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
    float alpha_accumulated = 0.0f;
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        float density = fmaxf(0.0f, densities[i]);
        float alpha = 1.0f - expf(-density * 0.01f); // Reduced step size
        float weight = alpha * (1.0f - alpha_accumulated);
        
        pixel_color[0] += weight * colors[i * 3 + 0];
        pixel_color[1] += weight * colors[i * 3 + 1];
        pixel_color[2] += weight * colors[i * 3 + 2];
        
        alpha_accumulated += weight;
        if (alpha_accumulated > 0.99f) break;
    }
}

// Compute gradients with gradient clipping
void compute_volume_rendering_gradients(float* mlp_output, float* pixel_error, 
                                       float* mlp_error_output) {
    float densities[NUM_SAMPLES];
    float colors[NUM_SAMPLES * 3];
    float alphas[NUM_SAMPLES];
    float transmittance[NUM_SAMPLES];
    float weights[NUM_SAMPLES];
    
    // Process MLP outputs with proper activations
    for (int s = 0; s < NUM_SAMPLES; s++) {
        densities[s] = fmaxf(0.0f, mlp_output[s * 4]);
        
        // Use sigmoid for colors instead of clamp
        colors[s * 3 + 0] = 1.0f / (1.0f + expf(-mlp_output[s * 4 + 1]));
        colors[s * 3 + 1] = 1.0f / (1.0f + expf(-mlp_output[s * 4 + 2]));
        colors[s * 3 + 2] = 1.0f / (1.0f + expf(-mlp_output[s * 4 + 3]));
        
        alphas[s] = 1.0f - expf(-densities[s] * 0.01f);
    }
    
    // Compute transmittance and weights
    float alpha_accumulated = 0.0f;
    for (int s = 0; s < NUM_SAMPLES; s++) {
        transmittance[s] = 1.0f - alpha_accumulated;
        weights[s] = alphas[s] * transmittance[s];
        alpha_accumulated += weights[s];
        if (alpha_accumulated > 0.99f) {
            // Zero out remaining weights
            for (int t = s + 1; t < NUM_SAMPLES; t++) {
                weights[t] = 0.0f;
            }
            break;
        }
    }
    
    // Compute gradients with clipping
    for (int s = 0; s < NUM_SAMPLES; s++) {
        // Color gradients with sigmoid derivative
        float color_gradients[3];
        for (int c = 0; c < 3; c++) {
            float sigmoid_val = colors[s * 3 + c];
            float sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
            color_gradients[c] = weights[s] * pixel_error[c] * sigmoid_deriv;
        }
        
        // Density gradient
        float density_gradient = 0.0f;
        
        if (mlp_output[s * 4] > 0.0f) { // ReLU derivative
            float dalpha_ddensity = 0.01f * expf(-densities[s] * 0.01f);
            
            // Direct contribution
            float dweight_ddensity = dalpha_ddensity * transmittance[s];
            density_gradient += dweight_ddensity * (
                pixel_error[0] * colors[s * 3 + 0] + 
                pixel_error[1] * colors[s * 3 + 1] + 
                pixel_error[2] * colors[s * 3 + 2]
            );
            
            // Indirect contribution
            for (int t = s + 1; t < NUM_SAMPLES; t++) {
                if (weights[t] > 1e-8f) { // Only if weight is significant
                    float dtransmittance_ddensity = -dalpha_ddensity * weights[s] / alphas[s];
                    float dweight_t_ddensity = alphas[t] * dtransmittance_ddensity;
                    density_gradient += dweight_t_ddensity * (
                        pixel_error[0] * colors[t * 3 + 0] + 
                        pixel_error[1] * colors[t * 3 + 1] + 
                        pixel_error[2] * colors[t * 3 + 2]
                    );
                }
            }
        }
        
        // Gradient clipping
        density_gradient = fmaxf(-10.0f, fminf(10.0f, density_gradient));
        for (int c = 0; c < 3; c++) {
            color_gradients[c] = fmaxf(-10.0f, fminf(10.0f, color_gradients[c]));
        }
        
        // Set gradients
        mlp_error_output[s * 4 + 0] = density_gradient;
        mlp_error_output[s * 4 + 1] = color_gradients[0];
        mlp_error_output[s * 4 + 2] = color_gradients[1];
        mlp_error_output[s * 4 + 3] = color_gradients[2];
    }
}

// Load training data and filter out pure black rays
int load_training_data(const char* filename, float** X, float** true_colors, int* num_rays) {
    FILE* csv = fopen(filename, "r");
    if (!csv) {
        fprintf(stderr, "Failed to open CSV file: %s\n", filename);
        return -1;
    }
    
    // Skip header and count lines
    char line[1024];
    fgets(line, sizeof(line), csv);
    
    int total_lines = 0;
    while (fgets(line, sizeof(line), csv)) {
        total_lines++;
    }
    
    int total_rays = total_lines / NUM_SAMPLES;
    printf("Found %d total rays\n", total_rays);
    
    // Temporary storage for all data
    float* temp_X = (float*)malloc(total_rays * NUM_SAMPLES * 6 * sizeof(float));
    float* temp_colors = (float*)malloc(total_rays * NUM_SAMPLES * 3 * sizeof(float));
    
    // Reset and read all data
    rewind(csv);
    fgets(line, sizeof(line), csv);
    
    int sample_index = 0;
    while (fgets(line, sizeof(line), csv) && sample_index < total_rays * NUM_SAMPLES) {
        char* token = strtok(line, ",");
        
        // Skip first 3 fields
        for (int i = 0; i < 3; i++) {
            token = strtok(NULL, ",");
        }
        
        // Read position and direction
        for (int i = 0; i < 6; i++) {
            temp_X[sample_index * 6 + i] = atof(token);
            token = strtok(NULL, ",");
        }
        
        // Read colors
        for (int i = 0; i < 3; i++) {
            temp_colors[sample_index * 3 + i] = atof(token);
            token = (i < 2) ? strtok(NULL, ",") : strtok(NULL, "\n");
        }
        
        sample_index++;
    }
    fclose(csv);
    
    // Filter out rays that are pure black
    int good_rays = 0;
    for (int ray = 0; ray < total_rays; ray++) {
        int ray_start = ray * NUM_SAMPLES;
        float max_color = 0.0f;
        
        // Check if this ray has any non-black pixels
        for (int s = 0; s < NUM_SAMPLES; s++) {
            int color_idx = (ray_start + s) * 3;
            float brightness = temp_colors[color_idx] + temp_colors[color_idx + 1] + temp_colors[color_idx + 2];
            if (brightness > max_color) max_color = brightness;
        }
        
        if (max_color > 0.01f) { // Keep rays with some color
            good_rays++;
        }
    }
    
    printf("Filtering to %d non-black rays\n", good_rays);
    *num_rays = good_rays;
    
    // Allocate final arrays
    *X = (float*)malloc(good_rays * NUM_SAMPLES * 6 * sizeof(float));
    *true_colors = (float*)malloc(good_rays * NUM_SAMPLES * 3 * sizeof(float));
    
    // Copy good rays
    int dest_ray = 0;
    for (int src_ray = 0; src_ray < total_rays && dest_ray < good_rays; src_ray++) {
        int src_start = src_ray * NUM_SAMPLES;
        float max_color = 0.0f;
        
        // Check brightness again
        for (int s = 0; s < NUM_SAMPLES; s++) {
            int color_idx = (src_start + s) * 3;
            float brightness = temp_colors[color_idx] + temp_colors[color_idx + 1] + temp_colors[color_idx + 2];
            if (brightness > max_color) max_color = brightness;
        }
        
        if (max_color > 0.01f) {
            int dest_start = dest_ray * NUM_SAMPLES;
            
            // Copy ray data
            memcpy(&(*X)[dest_start * 6], &temp_X[src_start * 6], NUM_SAMPLES * 6 * sizeof(float));
            memcpy(&(*true_colors)[dest_start * 3], &temp_colors[src_start * 3], NUM_SAMPLES * 3 * sizeof(float));
            
            dest_ray++;
        }
    }
    
    free(temp_X);
    free(temp_colors);
    return 0;
}

// Get random batch of rays
void get_random_batch(float* all_X, float* all_true_colors, int num_rays,
                     float* batch_X, float* batch_true_colors) {
    for (int ray = 0; ray < RAYS_PER_BATCH; ray++) {
        int random_ray = rand() % num_rays;
        int source_offset = random_ray * NUM_SAMPLES;
        int dest_offset = ray * NUM_SAMPLES;
        
        memcpy(&batch_X[dest_offset * 6], 
               &all_X[source_offset * 6], 
               NUM_SAMPLES * 6 * sizeof(float));
        
        memcpy(&batch_true_colors[dest_offset * 3], 
               &all_true_colors[source_offset * 3], 
               NUM_SAMPLES * 3 * sizeof(float));
    }
}

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);
    
    // Find CSV file
    char csv_filename[256];
    FILE* ls_pipe = popen("ls -t *.csv | head -1", "r");
    if (!ls_pipe || !fgets(csv_filename, sizeof(csv_filename), ls_pipe)) {
        fprintf(stderr, "No CSV file found\n");
        return -1;
    }
    csv_filename[strcspn(csv_filename, "\n")] = '\0';
    pclose(ls_pipe);
    
    printf("Using training data: %s\n", csv_filename);
    
    // Load filtered training data
    float *all_X, *all_true_colors;
    int num_rays;
    if (load_training_data(csv_filename, &all_X, &all_true_colors, &num_rays) != 0) {
        return -1;
    }
    
    // Network setup
    const int input_dim = 6;
    const int hidden_dim = 1024;
    const int output_dim = 4;
    const int batch_size = RAYS_PER_BATCH * NUM_SAMPLES;
    
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size);
    
    // Better weight initialization - scale up initial weights
    printf("Reinitializing weights with better scaling...\n");
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        mlp->W1[i] *= 2.0f; // Scale up input weights
    }
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        mlp->W2[i] *= 0.1f; // Scale down output weights to prevent saturation
    }
    for (int i = 0; i < output_dim * input_dim; i++) {
        mlp->W3[i] *= 0.1f; // Scale down skip connection
    }
    
    float* batch_X = (float*)malloc(batch_size * input_dim * sizeof(float));
    float* batch_true_colors = (float*)malloc(batch_size * 3 * sizeof(float));
    
    // Training parameters
    const int num_epochs = 20000;
    float learning_rate = 0.001f; // Start higher
    
    printf("Starting NeRF training...\n");
    printf("Architecture: %d -> %d -> %d\n", input_dim, hidden_dim, output_dim);
    printf("Filtered rays: %d\n", num_rays);
    printf("Initial learning rate: %.6f\n", learning_rate);
    printf("\n");
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Learning rate decay
        if (epoch > 0 && epoch % 5000 == 0) {
            learning_rate *= 0.5f;
            printf("Learning rate reduced to: %.6f\n", learning_rate);
        }
        
        get_random_batch(all_X, all_true_colors, num_rays, batch_X, batch_true_colors);
        
        forward_pass_mlp(mlp, batch_X);
        zero_gradients_mlp(mlp);
        
        float total_loss = 0.0f;
        
        for (int ray = 0; ray < RAYS_PER_BATCH; ray++) {
            int ray_start_idx = ray * NUM_SAMPLES;
            float* ray_mlp_output = &mlp->layer2_output[ray_start_idx * 4];
            float* ray_error_output = &mlp->error_output[ray_start_idx * 4];
            
            // Extract with sigmoid for colors
            float densities[NUM_SAMPLES];
            float colors[NUM_SAMPLES * 3];
            
            for (int s = 0; s < NUM_SAMPLES; s++) {
                densities[s] = fmaxf(0.0f, ray_mlp_output[s * 4]);
                colors[s * 3 + 0] = 1.0f / (1.0f + expf(-ray_mlp_output[s * 4 + 1]));
                colors[s * 3 + 1] = 1.0f / (1.0f + expf(-ray_mlp_output[s * 4 + 2]));
                colors[s * 3 + 2] = 1.0f / (1.0f + expf(-ray_mlp_output[s * 4 + 3]));
            }
            
            float predicted_pixel_color[3];
            render_ray(densities, colors, predicted_pixel_color);
            
            float* true_pixel_color = &batch_true_colors[ray_start_idx * 3];
            
            float pixel_error[3] = {
                predicted_pixel_color[0] - true_pixel_color[0],
                predicted_pixel_color[1] - true_pixel_color[1], 
                predicted_pixel_color[2] - true_pixel_color[2]
            };
            
            total_loss += (pixel_error[0] * pixel_error[0] + 
                          pixel_error[1] * pixel_error[1] + 
                          pixel_error[2] * pixel_error[2]);
            
            compute_volume_rendering_gradients(ray_mlp_output, pixel_error, ray_error_output);
        }
        
        backward_pass_mlp(mlp, batch_X);
        update_weights_mlp(mlp, learning_rate);
        
        if ((epoch + 1) % 100 == 0) {
            float average_loss = total_loss / RAYS_PER_BATCH;
            printf("Epoch [%d/%d], Loss: %.6f\n", epoch + 1, num_epochs, average_loss);
            
            if ((epoch + 1) % 1000 == 0) {
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
        }
    }
    
    // Save model
    char model_filename[64];
    time_t now = time(NULL);
    strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_nerf_model.bin", localtime(&now));
    save_mlp(mlp, model_filename);
    
    printf("Training completed! Model saved to: %s\n", model_filename);
    
    free(all_X);
    free(all_true_colors);
    free(batch_X);
    free(batch_true_colors);
    free_mlp(mlp);
    
    return 0;
}