#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cblas.h>
#include "mlp/mlp.h"

#define MAX_LINE_LENGTH 1024
#define NUM_RAYS 8
#define SAMPLES_PER_RAY 64
#define TOTAL_SAMPLES (NUM_RAYS * SAMPLES_PER_RAY)

typedef struct {
    char image_filename[256];
    int ray_id;
    int sample_id;
    float pos[3];    // x, y, z
    float dir[3];    // dx, dy, dz
    float color[3];  // r_true, g_true, b_true
} Sample;

// Volume rendering function - integrate along ray to get pixel color
void render_ray(float* densities, float* colors, int samples_per_ray, float* pixel_color) {
    // Simple volume rendering using alpha compositing
    float alpha_acc = 0.0f;
    pixel_color[0] = pixel_color[1] = pixel_color[2] = 0.0f;
    
    for (int i = 0; i < samples_per_ray; i++) {
        // Convert density to alpha (simplified)
        float density = fmaxf(0.0f, densities[i]);
        float alpha = 1.0f - expf(-density * 0.1f); // dt = 0.1 for simplicity
        
        // Alpha compositing
        float weight = alpha * (1.0f - alpha_acc);
        pixel_color[0] += weight * colors[i * 3 + 0];
        pixel_color[1] += weight * colors[i * 3 + 1];
        pixel_color[2] += weight * colors[i * 3 + 2];
        
        alpha_acc += weight;
        
        // Early termination if fully opaque
        if (alpha_acc > 0.99f) break;
    }
}

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);
    
    // Find the most recent CSV file
    char csv_filename[256] = "";
    FILE* ls = popen("ls -t *.csv | head -1", "r");
    if (ls && fgets(csv_filename, sizeof(csv_filename), ls)) {
        // Remove newline
        csv_filename[strcspn(csv_filename, "\n")] = 0;
        pclose(ls);
    } else {
        printf("No CSV file found!\n");
        if (ls) pclose(ls);
        return -1;
    }
    
    printf("Loading data from: %s\n", csv_filename);
    
    // Open CSV file
    FILE* csv = fopen(csv_filename, "r");
    if (!csv) {
        printf("Failed to open %s\n", csv_filename);
        return -1;
    }
    
    // Skip header line
    char line[MAX_LINE_LENGTH];
    fgets(line, sizeof(line), csv);
    
    // Load first 512 samples (8 rays × 64 samples)
    Sample samples[TOTAL_SAMPLES];
    int loaded_samples = 0;
    
    while (loaded_samples < TOTAL_SAMPLES && fgets(line, sizeof(line), csv)) {
        Sample* s = &samples[loaded_samples];
        
        // Parse CSV line
        char* token = strtok(line, ",");
        if (!token) continue;
        strcpy(s->image_filename, token);
        
        token = strtok(NULL, ","); if (!token) continue;
        s->ray_id = atoi(token);
        
        token = strtok(NULL, ","); if (!token) continue;
        s->sample_id = atoi(token);
        
        token = strtok(NULL, ","); if (!token) continue;
        s->pos[0] = atof(token);
        
        token = strtok(NULL, ","); if (!token) continue;
        s->pos[1] = atof(token);
        
        token = strtok(NULL, ","); if (!token) continue;
        s->pos[2] = atof(token);
        
        token = strtok(NULL, ","); if (!token) continue;
        s->dir[0] = atof(token);
        
        token = strtok(NULL, ","); if (!token) continue;
        s->dir[1] = atof(token);
        
        token = strtok(NULL, ","); if (!token) continue;
        s->dir[2] = atof(token);
        
        token = strtok(NULL, ","); if (!token) continue;
        s->color[0] = atof(token);
        
        token = strtok(NULL, ","); if (!token) continue;
        s->color[1] = atof(token);
        
        token = strtok(NULL, "\n"); if (!token) continue;
        s->color[2] = atof(token);
        
        loaded_samples++;
    }
    fclose(csv);
    
    if (loaded_samples != TOTAL_SAMPLES) {
        printf("Warning: Only loaded %d samples instead of %d\n", loaded_samples, TOTAL_SAMPLES);
    }
    
    printf("Loaded %d samples for %d rays\n", loaded_samples, NUM_RAYS);
    
    // Prepare input data for MLP
    // Input: [x, y, z, dx, dy, dz] = 6 dimensions
    // Output: [density, r, g, b] = 4 dimensions
    const int input_dim = 6;
    const int hidden_dim = 256;
    const int output_dim = 4;
    const int batch_size = loaded_samples;
    
    float* X = (float*)malloc(batch_size * input_dim * sizeof(float));
    
    // Fill input batch
    for (int i = 0; i < loaded_samples; i++) {
        X[i * input_dim + 0] = samples[i].pos[0];  // x
        X[i * input_dim + 1] = samples[i].pos[1];  // y
        X[i * input_dim + 2] = samples[i].pos[2];  // z
        X[i * input_dim + 3] = samples[i].dir[0];  // dx
        X[i * input_dim + 4] = samples[i].dir[1];  // dy
        X[i * input_dim + 5] = samples[i].dir[2];  // dz
    }
    
    // Initialize MLP
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size);
    
    printf("Initialized MLP with %d inputs, %d hidden units, %d outputs\n", 
           input_dim, hidden_dim, output_dim);
    
    // Forward pass through MLP
    printf("Running forward pass...\n");
    forward_pass_mlp(mlp, X);
    
    // Extract predictions
    float* predictions = mlp->layer2_output;
    
    printf("MLP forward pass complete. Sample predictions:\n");
    printf("Sample\tDensity\tR\tG\tB\tTrue RGB\n");
    printf("------------------------------------------------\n");
    
    for (int i = 0; i < 10 && i < loaded_samples; i++) {
        printf("%d\t%.3f\t%.3f\t%.3f\t%.3f\t(%.3f,%.3f,%.3f)\n", 
               i,
               predictions[i * output_dim + 0], // density
               predictions[i * output_dim + 1], // r
               predictions[i * output_dim + 2], // g
               predictions[i * output_dim + 3], // b
               samples[i].color[0], samples[i].color[1], samples[i].color[2]);
    }
    
    // Now render each ray to get predicted pixel colors
    printf("\nRendering rays to pixel colors...\n");
    printf("Ray\tPred RGB\t\tTrue RGB\n");
    printf("----------------------------------------\n");
    
    for (int ray = 0; ray < NUM_RAYS; ray++) {
        int start_idx = ray * SAMPLES_PER_RAY;
        
        // Extract densities and colors for this ray
        float ray_densities[SAMPLES_PER_RAY];
        float ray_colors[SAMPLES_PER_RAY * 3];
        
        for (int s = 0; s < SAMPLES_PER_RAY; s++) {
            int sample_idx = start_idx + s;
            ray_densities[s] = predictions[sample_idx * output_dim + 0];
            ray_colors[s * 3 + 0] = fmaxf(0.0f, fminf(1.0f, predictions[sample_idx * output_dim + 1])); // clamp R
            ray_colors[s * 3 + 1] = fmaxf(0.0f, fminf(1.0f, predictions[sample_idx * output_dim + 2])); // clamp G
            ray_colors[s * 3 + 2] = fmaxf(0.0f, fminf(1.0f, predictions[sample_idx * output_dim + 3])); // clamp B
        }
        
        // Render this ray
        float pred_pixel[3];
        render_ray(ray_densities, ray_colors, SAMPLES_PER_RAY, pred_pixel);
        
        // Get true color (same for all samples of this ray)
        float* true_color = samples[start_idx].color;
        
        printf("%d\t(%.3f,%.3f,%.3f)\t(%.3f,%.3f,%.3f)\n",
               ray,
               pred_pixel[0], pred_pixel[1], pred_pixel[2],
               true_color[0], true_color[1], true_color[2]);
    }
    
    printf("\nNeRF inference complete!\n");
    printf("Successfully processed %d rays with %d samples each through MLP\n", 
           NUM_RAYS, SAMPLES_PER_RAY);
    
    // Cleanup
    free(X);
    free_mlp(mlp);
    
    return 0;
}