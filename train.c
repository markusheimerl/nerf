#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "mlp/mlp.h"
#include "data.h"

#define RAYS_PER_BATCH 8

// Volume rendering: integrate densities and colors along ray
void render_ray(float* densities, float* colors, float* pixel_color) {
    pixel_color[0] = pixel_color[1] = pixel_color[2] = 0.0f;
    float transmittance = 1.0f;
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        float density = fmaxf(0.0f, densities[i]);
        float alpha = 1.0f - expf(-density * 0.01f);
        float weight = alpha * transmittance;
        
        pixel_color[0] += weight * colors[i * 3 + 0];
        pixel_color[1] += weight * colors[i * 3 + 1];
        pixel_color[2] += weight * colors[i * 3 + 2];
        
        transmittance *= (1.0f - alpha);
        if (transmittance < 0.01f) break;
    }
}

// Compute volume rendering gradients
void compute_volume_rendering_gradients(float* mlp_output, float* pixel_error, 
                                       float* mlp_error_output) {
    float densities[NUM_SAMPLES];
    float colors[NUM_SAMPLES * 3];
    float alphas[NUM_SAMPLES];
    float transmittance[NUM_SAMPLES];
    float weights[NUM_SAMPLES];
    
    // Process MLP outputs
    for (int s = 0; s < NUM_SAMPLES; s++) {
        densities[s] = fmaxf(0.0f, mlp_output[s * 4]);
        colors[s * 3 + 0] = 1.0f / (1.0f + expf(-mlp_output[s * 4 + 1]));
        colors[s * 3 + 1] = 1.0f / (1.0f + expf(-mlp_output[s * 4 + 2]));
        colors[s * 3 + 2] = 1.0f / (1.0f + expf(-mlp_output[s * 4 + 3]));
        alphas[s] = 1.0f - expf(-densities[s] * 0.01f);
    }
    
    // Compute transmittance and weights
    transmittance[0] = 1.0f;
    weights[0] = alphas[0] * transmittance[0];
    
    for (int s = 1; s < NUM_SAMPLES; s++) {
        transmittance[s] = transmittance[s-1] * (1.0f - alphas[s-1]);
        weights[s] = alphas[s] * transmittance[s];
    }
    
    // Compute gradients
    for (int s = 0; s < NUM_SAMPLES; s++) {
        // Color gradients with sigmoid derivative
        for (int c = 0; c < 3; c++) {
            float sigmoid_val = colors[s * 3 + c];
            float sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
            mlp_error_output[s * 4 + 1 + c] = weights[s] * pixel_error[c] * sigmoid_deriv;
        }
        
        // Density gradient
        float density_gradient = 0.0f;
        
        if (mlp_output[s * 4] > 0.0f) {
            float dalpha_ddensity = 0.01f * expf(-densities[s] * 0.01f);
            
            // Direct contribution
            float dweight_ddensity = dalpha_ddensity * transmittance[s];
            density_gradient += dweight_ddensity * (
                pixel_error[0] * colors[s * 3 + 0] + 
                pixel_error[1] * colors[s * 3 + 1] + 
                pixel_error[2] * colors[s * 3 + 2]
            );
            
            // Indirect contribution through transmittance
            for (int t = s + 1; t < NUM_SAMPLES; t++) {
                float dtransmittance_ddensity = -dalpha_ddensity;
                for (int k = s + 1; k <= t; k++) {
                    dtransmittance_ddensity *= (1.0f - alphas[k-1]);
                }
                
                float dweight_t_ddensity = alphas[t] * dtransmittance_ddensity;
                density_gradient += dweight_t_ddensity * (
                    pixel_error[0] * colors[t * 3 + 0] + 
                    pixel_error[1] * colors[t * 3 + 1] + 
                    pixel_error[2] * colors[t * 3 + 2]
                );
            }
        }
        
        mlp_error_output[s * 4 + 0] = density_gradient;
    }
}

// Process batch and compute loss
float process_batch(MLP* mlp, float* batch_true_colors) {
    float total_loss = 0.0f;
    
    for (int ray = 0; ray < RAYS_PER_BATCH; ray++) {
        int ray_start_idx = ray * NUM_SAMPLES;
        float* ray_mlp_output = &mlp->layer2_output[ray_start_idx * 4];
        float* ray_error_output = &mlp->error_output[ray_start_idx * 4];
        
        // Extract densities and colors with activations
        float densities[NUM_SAMPLES];
        float colors[NUM_SAMPLES * 3];
        
        for (int s = 0; s < NUM_SAMPLES; s++) {
            densities[s] = fmaxf(0.0f, ray_mlp_output[s * 4]);
            colors[s * 3 + 0] = 1.0f / (1.0f + expf(-ray_mlp_output[s * 4 + 1]));
            colors[s * 3 + 1] = 1.0f / (1.0f + expf(-ray_mlp_output[s * 4 + 2]));
            colors[s * 3 + 2] = 1.0f / (1.0f + expf(-ray_mlp_output[s * 4 + 3]));
        }
        
        // Render pixel
        float predicted_pixel_color[3];
        render_ray(densities, colors, predicted_pixel_color);
        
        // Compute pixel error
        float* true_pixel_color = &batch_true_colors[ray_start_idx * 3];
        float pixel_error[3] = {
            predicted_pixel_color[0] - true_pixel_color[0],
            predicted_pixel_color[1] - true_pixel_color[1], 
            predicted_pixel_color[2] - true_pixel_color[2]
        };
        
        // Accumulate loss
        total_loss += (pixel_error[0] * pixel_error[0] + 
                      pixel_error[1] * pixel_error[1] + 
                      pixel_error[2] * pixel_error[2]);
        
        // Compute gradients
        compute_volume_rendering_gradients(ray_mlp_output, pixel_error, ray_error_output);
    }
    
    return total_loss / RAYS_PER_BATCH;
}

// Print sample predictions
void print_sample_predictions(MLP* mlp, float* batch_true_colors) {
    printf("Sample predictions:\n");
    for (int ray = 0; ray < 3 && ray < RAYS_PER_BATCH; ray++) {
        int ray_start_idx = ray * NUM_SAMPLES;
        float* ray_output = &mlp->layer2_output[ray_start_idx * 4];
        
        float densities[NUM_SAMPLES], colors[NUM_SAMPLES * 3];
        for (int s = 0; s < NUM_SAMPLES; s++) {
            densities[s] = fmaxf(0.0f, ray_output[s * 4]);
            colors[s * 3 + 0] = 1.0f / (1.0f + expf(-ray_output[s * 4 + 1]));
            colors[s * 3 + 1] = 1.0f / (1.0f + expf(-ray_output[s * 4 + 2]));
            colors[s * 3 + 2] = 1.0f / (1.0f + expf(-ray_output[s * 4 + 3]));
        }
        
        float pixel_color[3];
        render_ray(densities, colors, pixel_color);
        float* true_color = &batch_true_colors[ray_start_idx * 3];
        
        printf("  Ray %d: pred(%.3f,%.3f,%.3f) true(%.3f,%.3f,%.3f)\n", 
               ray, pixel_color[0], pixel_color[1], pixel_color[2], 
               true_color[0], true_color[1], true_color[2]);
    }
    printf("\n");
}

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);

    // Load dataset
    Dataset* dataset = load_dataset("./data/transforms.json", "./data", 100);
    if (!dataset || dataset->num_images == 0) {
        fprintf(stderr, "Failed to load dataset\n");
        return -1;
    }

    // Network setup
    const int input_dim = 6;
    const int hidden_dim = 1024;
    const int output_dim = 4;
    const int batch_size = RAYS_PER_BATCH * NUM_SAMPLES;
    
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size);
    
    // Allocate batch memory
    float* batch_X = (float*)malloc(batch_size * input_dim * sizeof(float));
    float* batch_true_colors = (float*)malloc(batch_size * 3 * sizeof(float));
    
    // Training parameters
    const int num_batches = 20000;
    float learning_rate = 0.001f;

    // Training loop
    for (int batch = 0; batch < num_batches; batch++) {
        // Generate batch
        generate_random_batch(dataset, RAYS_PER_BATCH, batch_X, batch_true_colors);
        
        forward_pass_mlp(mlp, batch_X);
        zero_gradients_mlp(mlp);
        
        float loss = process_batch(mlp, batch_true_colors);
        
        backward_pass_mlp(mlp, batch_X);
        update_weights_mlp(mlp, learning_rate);
        
        // Print progress
        if ((batch + 1) % 100 == 0) {
            printf("Batch [%d/%d], Loss: %.6f\n", batch + 1, num_batches, loss);
            
            // Print sample predictions every 1000 batches
            if ((batch + 1) % 1000 == 0) {
                print_sample_predictions(mlp, batch_true_colors);
            }
        }
    }
    
    // Save model
    char model_filename[64];
    time_t now = time(NULL);
    strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    save_mlp(mlp, model_filename);
    
    printf("\nTraining completed! Model saved to: %s\n", model_filename);
    
    // Cleanup
    free_dataset(dataset);
    free(batch_X);
    free(batch_true_colors);
    free_mlp(mlp);
    
    return 0;
}