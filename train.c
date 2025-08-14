#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include "mlp/mlp.h"

// Volume rendering: integrate densities and colors along ray
void render_ray(float* densities, float* colors, float* pixel_color) {
    pixel_color[0] = pixel_color[1] = pixel_color[2] = 0.0f;
    float alpha_acc = 0.0f;
    
    for (int i = 0; i < 64; i++) {
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

int main() {
    openblas_set_num_threads(4);
    
    // Find CSV file
    char csv_filename[256];
    FILE* ls = popen("ls -t *.csv | head -1", "r");
    fgets(csv_filename, sizeof(csv_filename), ls);
    csv_filename[strcspn(csv_filename, "\n")] = 0;
    pclose(ls);
    
    FILE* csv = fopen(csv_filename, "r");
    char line[1024];
    fgets(line, sizeof(line), csv); // skip header
    
    // MLP setup
    const int batch_size = 512; // 8 rays × 64 samples
    const int input_dim = 6, output_dim = 4;
    MLP* mlp = init_mlp(input_dim, 256, output_dim, batch_size);
    float* X = (float*)malloc(batch_size * input_dim * sizeof(float));
    float* true_colors = (float*)malloc(batch_size * 3 * sizeof(float));
    
    int sample_idx = 0;
    int batch_count = 0;
    
    // Process batches
    while (fgets(line, sizeof(line), csv)) {
        // Parse: image,ray_id,sample_id,x,y,z,dx,dy,dz,r,g,b
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
            batch_count++;
            
            // Print every 1000th batch
            if (batch_count % 1000 == 0) {
                float* pred = mlp->layer2_output;
                printf("Batch %d:\n", batch_count);
                
                // Render each of the 8 rays
                for (int ray = 0; ray < 8; ray++) {
                    int start = ray * 64;
                    
                    // Extract densities and colors for this ray
                    float densities[64], colors[64 * 3];
                    for (int s = 0; s < 64; s++) {
                        densities[s] = pred[(start + s) * 4];
                        colors[s*3] = pred[(start + s) * 4 + 1];
                        colors[s*3+1] = pred[(start + s) * 4 + 2];
                        colors[s*3+2] = pred[(start + s) * 4 + 3];
                    }
                    
                    // Render pixel
                    float pixel[3];
                    render_ray(densities, colors, pixel);
                    
                    // True color (same for all samples in this ray)
                    float* true_rgb = &true_colors[start * 3];
                    
                    printf("  Ray %d: pred(%.3f,%.3f,%.3f) true(%.3f,%.3f,%.3f)\n", 
                           ray, pixel[0], pixel[1], pixel[2], 
                           true_rgb[0], true_rgb[1], true_rgb[2]);
                }
                printf("\n");
            }
            
            sample_idx = 0;
        }
    }
    
    printf("Processed %d batches total\n", batch_count);
    
    fclose(csv);
    free(X);
    free(true_colors);
    free_mlp(mlp);
    return 0;
}