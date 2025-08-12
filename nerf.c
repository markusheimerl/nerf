#include "nerf.h"

NeRF* init_nerf(int pos_encode_levels, int dir_encode_levels, int hidden_dim, int batch_size) {
    NeRF* nerf = (NeRF*)malloc(sizeof(NeRF));
    
    nerf->pos_encode_levels = pos_encode_levels;
    nerf->dir_encode_levels = dir_encode_levels;
    nerf->batch_size = batch_size;
    nerf->output_dim = 4; // RGB + density
    
    // Calculate input dimension after positional encoding
    // Position: 3 coordinates * (1 + 2 * pos_encode_levels) each
    // Direction: 3 coordinates * (1 + 2 * dir_encode_levels) each
    int pos_encoded_dim = 3 * (1 + 2 * pos_encode_levels);
    int dir_encoded_dim = 3 * (1 + 2 * dir_encode_levels);
    nerf->input_dim = pos_encoded_dim + dir_encoded_dim;
    
    // Initialize MLPs
    nerf->coarse_mlp = init_mlp(nerf->input_dim, hidden_dim, nerf->output_dim, batch_size);
    nerf->fine_mlp = NULL; // We'll keep it simple for now
    
    return nerf;
}

void free_nerf(NeRF* nerf) {
    if (nerf->coarse_mlp) free_mlp(nerf->coarse_mlp);
    if (nerf->fine_mlp) free_mlp(nerf->fine_mlp);
    free(nerf);
}

void positional_encoding(float* input, int input_dim, float* output, int levels) {
    int out_idx = 0;
    
    for (int i = 0; i < input_dim; i++) {
        // Original coordinate
        output[out_idx++] = input[i];
        
        // Sinusoidal encodings
        for (int l = 0; l < levels; l++) {
            float freq = powf(2.0f, l);
            output[out_idx++] = sinf(freq * PI * input[i]);
            output[out_idx++] = cosf(freq * PI * input[i]);
        }
    }
}

void generate_rays(Camera* camera, float* rays_o, float* rays_d, int* pixel_coords, int num_rays) {
    float half_width = camera->width * 0.5f;
    float half_height = camera->height * 0.5f;
    
    for (int i = 0; i < num_rays; i++) {
        int u = pixel_coords[i * 2];     // x pixel coordinate
        int v = pixel_coords[i * 2 + 1]; // y pixel coordinate
        
        // Convert pixel coordinates to normalized device coordinates
        float x = (u - half_width) / camera->focal;
        float y = -(v - half_height) / camera->focal; // Negative because image y goes down
        float z = -1.0f; // Looking down negative z-axis
        
        // Ray direction in camera space
        float dir_cam[3] = {x, y, z};
        
        // Normalize direction
        float norm = sqrtf(dir_cam[0]*dir_cam[0] + dir_cam[1]*dir_cam[1] + dir_cam[2]*dir_cam[2]);
        dir_cam[0] /= norm;
        dir_cam[1] /= norm;
        dir_cam[2] /= norm;
        
        // Transform to world space using camera rotation matrix
        rays_d[i*3 + 0] = camera->rotation[0]*dir_cam[0] + camera->rotation[1]*dir_cam[1] + camera->rotation[2]*dir_cam[2];
        rays_d[i*3 + 1] = camera->rotation[3]*dir_cam[0] + camera->rotation[4]*dir_cam[1] + camera->rotation[5]*dir_cam[2];
        rays_d[i*3 + 2] = camera->rotation[6]*dir_cam[0] + camera->rotation[7]*dir_cam[1] + camera->rotation[8]*dir_cam[2];
        
        // Ray origin is camera position
        rays_o[i*3 + 0] = camera->position[0];
        rays_o[i*3 + 1] = camera->position[1];
        rays_o[i*3 + 2] = camera->position[2];
    }
}

void sample_points_along_rays(float* rays_o, float* rays_d, int num_rays, float* points, float* directions, float* z_vals) {
    // Generate evenly spaced samples along each ray
    for (int r = 0; r < num_rays; r++) {
        for (int s = 0; s < NUM_SAMPLES; s++) {
            // Linear sampling between near and far planes
            float t = NEAR_PLANE + (FAR_PLANE - NEAR_PLANE) * s / (NUM_SAMPLES - 1);
            z_vals[r * NUM_SAMPLES + s] = t;
            
            // Point along ray: o + t * d
            int point_idx = (r * NUM_SAMPLES + s) * 3;
            points[point_idx + 0] = rays_o[r*3 + 0] + t * rays_d[r*3 + 0];
            points[point_idx + 1] = rays_o[r*3 + 1] + t * rays_d[r*3 + 1];
            points[point_idx + 2] = rays_o[r*3 + 2] + t * rays_d[r*3 + 2];
            
            // Direction is the same for all points along a ray
            directions[point_idx + 0] = rays_d[r*3 + 0];
            directions[point_idx + 1] = rays_d[r*3 + 1];
            directions[point_idx + 2] = rays_d[r*3 + 2];
        }
    }
}

void volume_render(float* colors, float* densities, float* z_vals, int num_rays, float* rgb_output) {
    for (int r = 0; r < num_rays; r++) {
        float accumulated_rgb[3] = {0.0f, 0.0f, 0.0f};
        float accumulated_alpha = 0.0f;
        
        for (int s = 0; s < NUM_SAMPLES; s++) {
            int sample_idx = r * NUM_SAMPLES + s;
            
            // Calculate distance between samples
            float delta;
            if (s == NUM_SAMPLES - 1) {
                delta = FAR_PLANE - z_vals[sample_idx];
            } else {
                delta = z_vals[sample_idx + 1] - z_vals[sample_idx];
            }
            
            // Convert density to alpha
            float alpha = 1.0f - expf(-densities[sample_idx] * delta);
            
            // Transmittance (how much light reaches this point)
            float transmittance = expf(-accumulated_alpha);
            
            // Accumulate color
            accumulated_rgb[0] += transmittance * alpha * colors[sample_idx * 3 + 0];
            accumulated_rgb[1] += transmittance * alpha * colors[sample_idx * 3 + 1];
            accumulated_rgb[2] += transmittance * alpha * colors[sample_idx * 3 + 2];
            
            // Accumulate alpha for transmittance calculation
            accumulated_alpha += densities[sample_idx] * delta;
        }
        
        // Output final RGB for this ray
        rgb_output[r * 3 + 0] = accumulated_rgb[0];
        rgb_output[r * 3 + 1] = accumulated_rgb[1];
        rgb_output[r * 3 + 2] = accumulated_rgb[2];
    }
}

void render_rays(NeRF* nerf, float* rays_o, float* rays_d, int num_rays, float* rgb_output) {
    // Allocate memory for sampling
    float* points = (float*)malloc(num_rays * NUM_SAMPLES * 3 * sizeof(float));
    float* directions = (float*)malloc(num_rays * NUM_SAMPLES * 3 * sizeof(float));
    float* z_vals = (float*)malloc(num_rays * NUM_SAMPLES * sizeof(float));
    
    // Sample points along rays
    sample_points_along_rays(rays_o, rays_d, num_rays, points, directions, z_vals);
    
    // Encode positions and directions
    int total_samples = num_rays * NUM_SAMPLES;
    float* encoded_input = (float*)malloc(total_samples * nerf->input_dim * sizeof(float));
    
    for (int i = 0; i < total_samples; i++) {
        float* pos_encoded = &encoded_input[i * nerf->input_dim];
        float* dir_encoded = pos_encoded + 3 * (1 + 2 * nerf->pos_encode_levels);
        
        // Encode position
        positional_encoding(&points[i * 3], 3, pos_encoded, nerf->pos_encode_levels);
        
        // Encode direction
        positional_encoding(&directions[i * 3], 3, dir_encoded, nerf->dir_encode_levels);
    }
    
    // Process in batches
    float* colors = (float*)malloc(total_samples * 3 * sizeof(float));
    float* densities = (float*)malloc(total_samples * sizeof(float));
    
    for (int batch_start = 0; batch_start < total_samples; batch_start += nerf->batch_size) {
        int batch_end = (batch_start + nerf->batch_size < total_samples) ? 
                        batch_start + nerf->batch_size : total_samples;
        int current_batch_size = batch_end - batch_start;
        
        // Forward pass through MLP
        if (current_batch_size == nerf->batch_size) {
            forward_pass_mlp(nerf->coarse_mlp, &encoded_input[batch_start * nerf->input_dim]);
            
            // Extract colors and densities from MLP output
            for (int i = 0; i < current_batch_size; i++) {
                int global_idx = batch_start + i;
                colors[global_idx * 3 + 0] = nerf->coarse_mlp->layer2_output[i * 4 + 0]; // R
                colors[global_idx * 3 + 1] = nerf->coarse_mlp->layer2_output[i * 4 + 1]; // G
                colors[global_idx * 3 + 2] = nerf->coarse_mlp->layer2_output[i * 4 + 2]; // B
                densities[global_idx] = fmaxf(0.0f, nerf->coarse_mlp->layer2_output[i * 4 + 3]); // σ (ReLU)
            }
        }
    }
    
    // Apply sigmoid to colors to ensure they're in [0,1]
    for (int i = 0; i < total_samples * 3; i++) {
        colors[i] = 1.0f / (1.0f + expf(-colors[i]));
    }
    
    // Volume rendering
    volume_render(colors, densities, z_vals, num_rays, rgb_output);
    
    // Cleanup
    free(points);
    free(directions);
    free(z_vals);
    free(encoded_input);
    free(colors);
    free(densities);
}

// Save NeRF model
void save_nerf(NeRF* nerf, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save NeRF parameters
    fwrite(&nerf->pos_encode_levels, sizeof(int), 1, file);
    fwrite(&nerf->dir_encode_levels, sizeof(int), 1, file);
    fwrite(&nerf->input_dim, sizeof(int), 1, file);
    fwrite(&nerf->output_dim, sizeof(int), 1, file);
    fwrite(&nerf->batch_size, sizeof(int), 1, file);
    
    fclose(file);
    
    // Save the MLP separately
    char mlp_filename[256];
    snprintf(mlp_filename, sizeof(mlp_filename), "%s_coarse.bin", filename);
    save_mlp(nerf->coarse_mlp, mlp_filename);
    
    printf("NeRF saved to %s\n", filename);
}

// Load NeRF model
NeRF* load_nerf(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Load NeRF parameters
    int pos_encode_levels, dir_encode_levels, input_dim, output_dim, stored_batch_size;
    fread(&pos_encode_levels, sizeof(int), 1, file);
    fread(&dir_encode_levels, sizeof(int), 1, file);
    fread(&input_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    fclose(file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize NeRF structure
    NeRF* nerf = (NeRF*)malloc(sizeof(NeRF));
    nerf->pos_encode_levels = pos_encode_levels;
    nerf->dir_encode_levels = dir_encode_levels;
    nerf->input_dim = input_dim;
    nerf->output_dim = output_dim;
    nerf->batch_size = batch_size;
    
    // Load the MLP
    char mlp_filename[256];
    snprintf(mlp_filename, sizeof(mlp_filename), "%s_coarse.bin", filename);
    nerf->coarse_mlp = load_mlp(mlp_filename, batch_size);
    nerf->fine_mlp = NULL;
    
    printf("NeRF loaded from %s\n", filename);
    return nerf;
}