#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "mlp/mlp.h"

#define NUM_SAMPLES 64
#define RAYS_PER_BATCH 8

// Volume rendering: integrate densities and colors along ray
void render_ray(float* densities, float* colors, float* pixel_color) {
    pixel_color[0] = pixel_color[1] = pixel_color[2] = 0.0f;
    float alpha_acc = 0.0f;
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        float density = fmaxf(0.0f, densities[i]);
        float alpha = 1.0f - expf(-density * 0.1f);
        float weight = alpha * (1.0f - alpha_acc);
        
        pixel_color[0] += weight * fmaxf(0, fminf(1, colors[i * 3 + 0]));
        pixel_color[1] += weight * fmaxf(0, fminf(1, colors[i * 3 + 1]));
        pixel_color[2] += weight * fmaxf(0, fminf(1, colors[i * 3 + 2]));
        
        alpha_acc += weight;
        if (alpha_acc > 0.99f) break;
    }
}

// Compute gradients through volume rendering and set them directly in MLP error_output
void compute_volume_rendering_gradients(float* mlp_output, float* pixel_error, 
                                       float* mlp_error_output) {
    // Extract densities and colors
    float densities[NUM_SAMPLES];
    float colors[NUM_SAMPLES * 3];
    float alphas[NUM_SAMPLES];
    float transmittance[NUM_SAMPLES];
    float weights[NUM_SAMPLES];
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        densities[s] = fmaxf(0.0f, mlp_output[s * 4]); // ReLU applied HERE in volume rendering
        colors[s*3] = fmaxf(0, fminf(1, mlp_output[s * 4 + 1]));
        colors[s*3+1] = fmaxf(0, fminf(1, mlp_output[s * 4 + 2]));
        colors[s*3+2] = fmaxf(0, fminf(1, mlp_output[s * 4 + 3]));
        alphas[s] = 1.0f - expf(-densities[s] * 0.1f);
    }
    
    // Compute transmittance and weights
    float alpha_acc = 0.0f;
    for (int s = 0; s < NUM_SAMPLES; s++) {
        transmittance[s] = 1.0f - alpha_acc;
        weights[s] = alphas[s] * transmittance[s];
        alpha_acc += weights[s];
        if (alpha_acc > 0.99f) break;
    }
    
    // Compute gradients and set directly in MLP error_output
    for (int s = 0; s < NUM_SAMPLES; s++) {
        // Color gradients (simple)
        float color_grads[3] = {
            weights[s] * pixel_error[0],
            weights[s] * pixel_error[1], 
            weights[s] * pixel_error[2]
        };
        
        // Apply derivative of clamp function for colors
        if (mlp_output[s * 4 + 1] <= 0.0f || mlp_output[s * 4 + 1] >= 1.0f) color_grads[0] = 0.0f;
        if (mlp_output[s * 4 + 2] <= 0.0f || mlp_output[s * 4 + 2] >= 1.0f) color_grads[1] = 0.0f;
        if (mlp_output[s * 4 + 3] <= 0.0f || mlp_output[s * 4 + 3] >= 1.0f) color_grads[2] = 0.0f;
        
        // Density gradient (complex due to transmittance coupling)
        float density_grad = 0.0f;
        
        // Only compute gradient if MLP output is positive (derivative of ReLU in volume rendering)
        if (mlp_output[s * 4] > 0.0f) {
            float dalpha_ddensity = 0.1f * expf(-densities[s] * 0.1f);
            
            // Direct contribution
            float dweight_ddensity = dalpha_ddensity * transmittance[s];
            density_grad += dweight_ddensity * (pixel_error[0] * colors[s*3] + 
                                              pixel_error[1] * colors[s*3+1] + 
                                              pixel_error[2] * colors[s*3+2]);
            
            // Indirect contribution on subsequent samples
            for (int t = s + 1; t < NUM_SAMPLES; t++) {
                float dtransmittance_ddensity = -dalpha_ddensity * transmittance[s];
                float dweight_t_ddensity = alphas[t] * dtransmittance_ddensity;
                density_grad += dweight_t_ddensity * (pixel_error[0] * colors[t*3] + 
                                                    pixel_error[1] * colors[t*3+1] + 
                                                    pixel_error[2] * colors[t*3+2]);
            }
        }
        
        // Set gradients in MLP error_output
        mlp_error_output[s * 4] = density_grad;
        mlp_error_output[s * 4 + 1] = color_grads[0];
        mlp_error_output[s * 4 + 2] = color_grads[1];
        mlp_error_output[s * 4 + 3] = color_grads[2];
    }
}

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);
    
    // Find CSV file
    char csv_filename[256];
    FILE* ls = popen("ls -t *.csv | head -1", "r");
    if (!ls || !fgets(csv_filename, sizeof(csv_filename), ls)) {
        fprintf(stderr, "No CSV file found\n");
        return -1;
    }
    csv_filename[strcspn(csv_filename, "\n")] = 0;
    pclose(ls);
    
    printf("Using data file: %s\n", csv_filename);
    
    FILE* csv = fopen(csv_filename, "r");
    if (!csv) {
        fprintf(stderr, "Failed to open CSV file\n");
        return -1;
    }
    
    char line[1024];
    fgets(line, sizeof(line), csv); // skip header
    
    // MLP setup
    const int batch_size = RAYS_PER_BATCH * NUM_SAMPLES;
    const int input_dim = 6, output_dim = 4;
    MLP* mlp = init_mlp(input_dim, 256, output_dim, batch_size);
    
    float* X = (float*)malloc(batch_size * input_dim * sizeof(float));
    float* true_colors = (float*)malloc(batch_size * 3 * sizeof(float));
    
    int sample_idx = 0;
    int batch_count = 0;
    const float learning_rate = 0.0003f;
    
    printf("Starting NeRF training...\n");
    
    while (fgets(line, sizeof(line), csv)) {
        // Parse CSV line
        char* tok = strtok(line, ",");
        for (int i = 0; i < 3; i++) tok = strtok(NULL, ","); // skip first 3 fields
        
        // Get x,y,z,dx,dy,dz
        for (int i = 0; i < 6; i++) {
            X[sample_idx * 6 + i] = atof(tok);
            tok = strtok(NULL, ",");
        }
        
        // Get r,g,b
        for (int i = 0; i < 3; i++) {
            true_colors[sample_idx * 3 + i] = atof(tok);
            tok = (i < 2) ? strtok(NULL, ",") : strtok(NULL, "\n");
        }
        
        sample_idx++;
        
        if (sample_idx == batch_size) {
            // Forward pass
            forward_pass_mlp(mlp, X);
            
            // Zero gradients and compute volume rendering gradients
            zero_gradients_mlp(mlp);
            float total_loss = 0.0f;
            
            for (int ray = 0; ray < RAYS_PER_BATCH; ray++) {
                int start = ray * NUM_SAMPLES;
                float* ray_mlp_output = &mlp->layer2_output[start * 4];
                float* ray_error_output = &mlp->error_output[start * 4];
                
                // Volume render this ray
                float densities[NUM_SAMPLES], colors[NUM_SAMPLES * 3];
                for (int s = 0; s < NUM_SAMPLES; s++) {
                    densities[s] = fmaxf(0.0f, ray_mlp_output[s * 4]);
                    colors[s*3] = fmaxf(0, fminf(1, ray_mlp_output[s * 4 + 1]));
                    colors[s*3+1] = fmaxf(0, fminf(1, ray_mlp_output[s * 4 + 2]));
                    colors[s*3+2] = fmaxf(0, fminf(1, ray_mlp_output[s * 4 + 3]));
                }
                
                float pixel_color[3];
                render_ray(densities, colors, pixel_color);
                
                // Compute pixel error
                float* true_rgb = &true_colors[start * 3];
                float pixel_error[3] = {
                    pixel_color[0] - true_rgb[0],
                    pixel_color[1] - true_rgb[1], 
                    pixel_color[2] - true_rgb[2]
                };
                
                total_loss += pixel_error[0] * pixel_error[0] + 
                             pixel_error[1] * pixel_error[1] + 
                             pixel_error[2] * pixel_error[2];
                
                // Compute and set volume rendering gradients directly
                compute_volume_rendering_gradients(ray_mlp_output, pixel_error, ray_error_output);
            }
            
            // Backward pass and update (gradients already set in error_output)
            backward_pass_mlp(mlp, X);
            update_weights_mlp(mlp, learning_rate);
            
            batch_count++;
            
            // Print progress
            if (batch_count % 100 == 0) {
                printf("Batch %d: Loss: %.6f\n", batch_count, total_loss / RAYS_PER_BATCH);
                
                if (batch_count % 5000 == 0) {
                    for (int ray = 0; ray < 3; ray++) {
                        int start = ray * NUM_SAMPLES;
                        float* ray_output = &mlp->layer2_output[start * 4];
                        
                        float densities[NUM_SAMPLES], colors[NUM_SAMPLES * 3];
                        for (int s = 0; s < NUM_SAMPLES; s++) {
                            densities[s] = fmaxf(0.0f, ray_output[s * 4]);
                            colors[s*3] = fmaxf(0, fminf(1, ray_output[s * 4 + 1]));
                            colors[s*3+1] = fmaxf(0, fminf(1, ray_output[s * 4 + 2]));
                            colors[s*3+2] = fmaxf(0, fminf(1, ray_output[s * 4 + 3]));
                        }
                        
                        float pixel[3];
                        render_ray(densities, colors, pixel);
                        float* true_rgb = &true_colors[start * 3];
                        
                        printf("  Ray %d: pred(%.3f,%.3f,%.3f) true(%.3f,%.3f,%.3f)\n", 
                               ray, pixel[0], pixel[1], pixel[2], true_rgb[0], true_rgb[1], true_rgb[2]);
                    }
                    printf("\n");
                }
            }
            
            sample_idx = 0;
        }
    }
    
    // Save model
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_nerf_model.bin", localtime(&now));
    save_mlp(mlp, model_fname);
    
    printf("Training completed! Processed %d batches\n", batch_count);
    printf("Model saved to: %s\n", model_fname);
    
    fclose(csv);
    free(X);
    free(true_colors);
    free_mlp(mlp);
    return 0;
}