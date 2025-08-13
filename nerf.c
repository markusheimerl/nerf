#include "nerf.h"
#include <json-c/json.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

// CUDA kernel for ray generation
__global__ void generate_rays_kernel(float* rays_o, float* rays_d, int* pixel_coords, 
                                    float* cam_pos, float* cam_rot, float focal, 
                                    float half_width, float half_height, int num_rays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rays) {
        int u = pixel_coords[idx * 2];
        int v = pixel_coords[idx * 2 + 1];
        
        // Convert to normalized device coordinates
        float x = (u - half_width) / focal;
        float y = -(v - half_height) / focal;
        float z = -1.0f;
        
        // Normalize direction
        float norm = sqrtf(x*x + y*y + z*z);
        x /= norm; y /= norm; z /= norm;
        
        // Transform to world space
        rays_d[idx*3 + 0] = cam_rot[0]*x + cam_rot[1]*y + cam_rot[2]*z;
        rays_d[idx*3 + 1] = cam_rot[3]*x + cam_rot[4]*y + cam_rot[5]*z;
        rays_d[idx*3 + 2] = cam_rot[6]*x + cam_rot[7]*y + cam_rot[8]*z;
        
        // Ray origin is camera position
        rays_o[idx*3 + 0] = cam_pos[0];
        rays_o[idx*3 + 1] = cam_pos[1];
        rays_o[idx*3 + 2] = cam_pos[2];
    }
}

// CUDA kernel for sampling points along rays
__global__ void sample_points_kernel(float* rays_o, float* rays_d, float* points, 
                                    float* directions, float* z_vals, int num_rays) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ray_idx < num_rays && sample_idx < NUM_SAMPLES) {
        float t = NEAR_PLANE + (FAR_PLANE - NEAR_PLANE) * sample_idx / (NUM_SAMPLES - 1);
        int global_idx = ray_idx * NUM_SAMPLES + sample_idx;
        
        z_vals[global_idx] = t;
        
        // Point along ray: o + t * d
        points[global_idx * 3 + 0] = rays_o[ray_idx*3 + 0] + t * rays_d[ray_idx*3 + 0];
        points[global_idx * 3 + 1] = rays_o[ray_idx*3 + 1] + t * rays_d[ray_idx*3 + 1];
        points[global_idx * 3 + 2] = rays_o[ray_idx*3 + 2] + t * rays_d[ray_idx*3 + 2];
        
        // Direction is the same for all points along a ray
        directions[global_idx * 3 + 0] = rays_d[ray_idx*3 + 0];
        directions[global_idx * 3 + 1] = rays_d[ray_idx*3 + 1];
        directions[global_idx * 3 + 2] = rays_d[ray_idx*3 + 2];
    }
}

// CUDA kernel for positional encoding
__global__ void positional_encoding_kernel(float* input, float* output, int input_dim, 
                                          int levels, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        int output_dim = input_dim * (1 + 2 * levels);
        
        for (int i = 0; i < input_dim; i++) {
            int out_idx = idx * output_dim + i * (1 + 2 * levels);
            
            // Original coordinate
            output[out_idx] = input[idx * input_dim + i];
            out_idx++;
            
            // Sinusoidal encodings
            for (int l = 0; l < levels; l++) {
                float freq = powf(2.0f, l);
                float val = freq * PI * input[idx * input_dim + i];
                output[out_idx++] = sinf(val);
                output[out_idx++] = cosf(val);
            }
        }
    }
}

// CUDA kernel for volume rendering
__global__ void volume_render_kernel(float* colors, float* densities, float* z_vals, 
                                    float* rgb_output, int num_rays) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx < num_rays) {
        float accumulated_rgb[3] = {0.0f, 0.0f, 0.0f};
        float accumulated_alpha = 0.0f;
        
        for (int s = 0; s < NUM_SAMPLES; s++) {
            int sample_idx = ray_idx * NUM_SAMPLES + s;
            
            // Calculate distance between samples
            float delta;
            if (s == NUM_SAMPLES - 1) {
                delta = FAR_PLANE - z_vals[sample_idx];
            } else {
                delta = z_vals[sample_idx + 1] - z_vals[sample_idx];
            }
            
            // Convert density to alpha
            float alpha = 1.0f - expf(-densities[sample_idx] * delta);
            
            // Transmittance
            float transmittance = expf(-accumulated_alpha);
            
            // Accumulate color
            accumulated_rgb[0] += transmittance * alpha * colors[sample_idx * 3 + 0];
            accumulated_rgb[1] += transmittance * alpha * colors[sample_idx * 3 + 1];
            accumulated_rgb[2] += transmittance * alpha * colors[sample_idx * 3 + 2];
            
            // Accumulate alpha
            accumulated_alpha += densities[sample_idx] * delta;
        }
        
        rgb_output[ray_idx * 3 + 0] = accumulated_rgb[0];
        rgb_output[ray_idx * 3 + 1] = accumulated_rgb[1];
        rgb_output[ray_idx * 3 + 2] = accumulated_rgb[2];
    }
}

// CUDA kernel for volume rendering backward pass
__global__ void volume_render_backward_kernel(
    float* colors, float* densities, float* z_vals,
    float* rgb_grad, float* color_grad, float* density_grad,
    int num_rays) {
    
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx < num_rays) {
        
        // Forward pass to compute transmittances
        float accumulated_alpha = 0.0f;
        float transmittances[NUM_SAMPLES];
        float alphas[NUM_SAMPLES];
        
        for (int s = 0; s < NUM_SAMPLES; s++) {
            int sample_idx = ray_idx * NUM_SAMPLES + s;
            
            float delta;
            if (s == NUM_SAMPLES - 1) {
                delta = FAR_PLANE - z_vals[sample_idx];
            } else {
                delta = z_vals[sample_idx + 1] - z_vals[sample_idx];
            }
            
            alphas[s] = 1.0f - expf(-densities[sample_idx] * delta);
            transmittances[s] = expf(-accumulated_alpha);
            accumulated_alpha += densities[sample_idx] * delta;
        }
        
        // Backward pass
        for (int s = 0; s < NUM_SAMPLES; s++) {
            int sample_idx = ray_idx * NUM_SAMPLES + s;
            
            float delta;
            if (s == NUM_SAMPLES - 1) {
                delta = FAR_PLANE - z_vals[sample_idx];
            } else {
                delta = z_vals[sample_idx + 1] - z_vals[sample_idx];
            }
            
            // Gradient w.r.t colors
            for (int c = 0; c < 3; c++) {
                color_grad[sample_idx * 3 + c] = 
                    rgb_grad[ray_idx * 3 + c] * transmittances[s] * alphas[s];
            }
            
            // Gradient w.r.t density
            float density_grad_val = 0.0f;
            for (int c = 0; c < 3; c++) {
                // Direct contribution
                density_grad_val += rgb_grad[ray_idx * 3 + c] * 
                    transmittances[s] * expf(-densities[sample_idx] * delta) * delta * 
                    colors[sample_idx * 3 + c];
                
                // Indirect contribution through transmittance
                for (int s2 = s + 1; s2 < NUM_SAMPLES; s2++) {
                    int sample_idx2 = ray_idx * NUM_SAMPLES + s2;
                    density_grad_val -= rgb_grad[ray_idx * 3 + c] *
                        transmittances[s2] * alphas[s2] * delta *
                        colors[sample_idx2 * 3 + c];
                }
            }
            density_grad[sample_idx] = density_grad_val;
        }
    }
}

// Utility kernels
__global__ void sigmoid_activation_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

__global__ void relu_activation_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void mse_loss_kernel(float* rendered, float* target, float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = rendered[idx] - target[idx];
        atomicAdd(loss, diff * diff);
    }
}

__global__ void mse_grad_kernel(float* rendered, float* target, float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 2.0f * (rendered[idx] - target[idx]);
    }
}

NeRF* init_nerf(int pos_encode_levels, int dir_encode_levels, int hidden_dim, 
                int batch_size, int max_rays, cublasHandle_t cublas_handle) {
    NeRF* nerf = (NeRF*)malloc(sizeof(NeRF));
    
    nerf->pos_encode_levels = pos_encode_levels;
    nerf->dir_encode_levels = dir_encode_levels;
    nerf->batch_size = batch_size;
    nerf->output_dim = 4; // RGB + density
    nerf->cublas_handle = cublas_handle;
    
    // Calculate input dimension after positional encoding
    int pos_encoded_dim = 3 * (1 + 2 * pos_encode_levels);
    int dir_encoded_dim = 3 * (1 + 2 * dir_encode_levels);
    nerf->input_dim = pos_encoded_dim + dir_encoded_dim;
    
    // Initialize MLP
    nerf->coarse_mlp = init_mlp(nerf->input_dim, hidden_dim, nerf->output_dim, batch_size, cublas_handle);
    nerf->fine_mlp = NULL;
    
    // Allocate device memory for ray processing
    int max_samples = max_rays * NUM_SAMPLES;
    CHECK_CUDA(cudaMalloc(&nerf->d_rays_o, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rays_d, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_points, max_samples * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_directions, max_samples * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_z_vals, max_samples * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_encoded_input, max_samples * nerf->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_colors, max_samples * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_densities, max_samples * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rendered_rgb, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_target_rgb, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_color_grad, max_samples * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_density_grad, max_samples * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rgb_grad, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_pixel_coords, max_rays * 2 * sizeof(int)));
    
    return nerf;
}

void free_nerf(NeRF* nerf) {
    if (nerf->coarse_mlp) free_mlp(nerf->coarse_mlp);
    if (nerf->fine_mlp) free_mlp(nerf->fine_mlp);
    
    cudaFree(nerf->d_rays_o);
    cudaFree(nerf->d_rays_d);
    cudaFree(nerf->d_points);
    cudaFree(nerf->d_directions);
    cudaFree(nerf->d_z_vals);
    cudaFree(nerf->d_encoded_input);
    cudaFree(nerf->d_colors);
    cudaFree(nerf->d_densities);
    cudaFree(nerf->d_rendered_rgb);
    cudaFree(nerf->d_target_rgb);
    cudaFree(nerf->d_color_grad);
    cudaFree(nerf->d_density_grad);
    cudaFree(nerf->d_rgb_grad);
    cudaFree(nerf->d_pixel_coords);
    
    free(nerf);
}

void render_rays(NeRF* nerf, float* d_rays_o, float* d_rays_d, int num_rays) {
    int block_size = 256;
    int num_blocks;
    
    // Sample points along rays
    dim3 block_dim(16, 4);
    dim3 grid_dim((num_rays + block_dim.x - 1) / block_dim.x, 
                  (NUM_SAMPLES + block_dim.y - 1) / block_dim.y);
    sample_points_kernel<<<grid_dim, block_dim>>>(
        d_rays_o, d_rays_d, nerf->d_points, nerf->d_directions, nerf->d_z_vals, num_rays);
    
    int total_samples = num_rays * NUM_SAMPLES;
    
    // Encode positions
    num_blocks = (total_samples + block_size - 1) / block_size;
    positional_encoding_kernel<<<num_blocks, block_size>>>(
        nerf->d_points, nerf->d_encoded_input, 3, nerf->pos_encode_levels, total_samples);
    
    // Encode directions (offset in encoded_input)
    int pos_encoded_dim = 3 * (1 + 2 * nerf->pos_encode_levels);
    float* dir_encoded_start = nerf->d_encoded_input;
    for (int i = 0; i < total_samples; i++) {
        dir_encoded_start += pos_encoded_dim;
    }
    positional_encoding_kernel<<<num_blocks, block_size>>>(
        nerf->d_directions, dir_encoded_start, 3, nerf->dir_encode_levels, total_samples);
    
    // Process through MLP in batches
    for (int batch_start = 0; batch_start < total_samples; batch_start += nerf->batch_size) {
        int batch_end = min(batch_start + nerf->batch_size, total_samples);
        int current_batch_size = batch_end - batch_start;
        
        if (current_batch_size == nerf->batch_size) {
            // Forward pass
            forward_pass_mlp(nerf->coarse_mlp, &nerf->d_encoded_input[batch_start * nerf->input_dim]);
            
            // Extract colors and densities
            CHECK_CUBLAS(cublasScopy(nerf->cublas_handle, current_batch_size * 3,
                                   nerf->coarse_mlp->d_layer2_output, 1,
                                   &nerf->d_colors[batch_start * 3], 1));
            
            // Copy densities (4th output)
            for (int i = 0; i < current_batch_size; i++) {
                CHECK_CUDA(cudaMemcpy(&nerf->d_densities[batch_start + i],
                                     &nerf->coarse_mlp->d_layer2_output[i * 4 + 3],
                                     sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }
    }
    
    // Apply activations
    num_blocks = (total_samples * 3 + block_size - 1) / block_size;
    sigmoid_activation_kernel<<<num_blocks, block_size>>>(nerf->d_colors, total_samples * 3);
    
    num_blocks = (total_samples + block_size - 1) / block_size;
    relu_activation_kernel<<<num_blocks, block_size>>>(nerf->d_densities, total_samples);
    
    // Volume rendering
    num_blocks = (num_rays + block_size - 1) / block_size;
    volume_render_kernel<<<num_blocks, block_size>>>(
        nerf->d_colors, nerf->d_densities, nerf->d_z_vals, nerf->d_rendered_rgb, num_rays);
}

float calculate_loss_nerf(NeRF* nerf, float* d_target_rgb, int num_rays) {
    float* d_loss;
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
    
    int block_size = 256;
    int num_blocks = (num_rays * 3 + block_size - 1) / block_size;
    mse_loss_kernel<<<num_blocks, block_size>>>(
        nerf->d_rendered_rgb, d_target_rgb, d_loss, num_rays * 3);
    
    float h_loss;
    CHECK_CUDA(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_loss));
    
    return h_loss / (num_rays * 3);
}

void backward_pass_nerf(NeRF* nerf, float* d_target_rgb, int num_rays) {
    int block_size = 256;
    int num_blocks;
    
    // Calculate gradients w.r.t rendered RGB
    num_blocks = (num_rays * 3 + block_size - 1) / block_size;
    mse_grad_kernel<<<num_blocks, block_size>>>(
        nerf->d_rendered_rgb, d_target_rgb, nerf->d_rgb_grad, num_rays * 3);
    
    // Backward pass through volume rendering
    num_blocks = (num_rays + block_size - 1) / block_size;
    volume_render_backward_kernel<<<num_blocks, block_size>>>(
        nerf->d_colors, nerf->d_densities, nerf->d_z_vals,
        nerf->d_rgb_grad, nerf->d_color_grad, nerf->d_density_grad, num_rays);
    
    // TODO: Backward pass through MLP using color_grad and density_grad
    // This would require implementing backward pass for the MLP with custom gradients
    // For now, we'll use the simplified approach of just updating MLP weights
}

void update_weights_nerf(NeRF* nerf, float learning_rate) {
    update_weights_mlp(nerf->coarse_mlp, learning_rate);
}

Camera* load_camera_from_transforms(const char* filename, int frame_idx) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error: Could not open %s\n", filename);
        return NULL;
    }
    
    // Read entire file
    fseek(fp, 0, SEEK_END);
    long length = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* data = (char*)malloc(length + 1);
    fread(data, 1, length, fp);
    data[length] = '\0';
    fclose(fp);
    
    // Parse JSON
    json_object* root = json_tokener_parse(data);
    json_object* camera_angle_x_obj;
    json_object* frames_obj;
    
    if (!json_object_object_get_ex(root, "camera_angle_x", &camera_angle_x_obj) ||
        !json_object_object_get_ex(root, "frames", &frames_obj)) {
        printf("Error: Invalid transforms.json format\n");
        free(data);
        json_object_put(root);
        return NULL;
    }
    
    double camera_angle_x = json_object_get_double(camera_angle_x_obj);
    int num_frames = json_object_array_length(frames_obj);
    
    if (frame_idx >= num_frames) {
        printf("Error: Frame index %d out of range (0-%d)\n", frame_idx, num_frames - 1);
        free(data);
        json_object_put(root);
        return NULL;
    }
    
    json_object* frame = json_object_array_get_idx(frames_obj, frame_idx);
    json_object* transform_matrix_obj;
    
    if (!json_object_object_get_ex(frame, "transform_matrix", &transform_matrix_obj)) {
        printf("Error: No transform_matrix in frame %d\n", frame_idx);
        free(data);
        json_object_put(root);
        return NULL;
    }
    
    Camera* cam = (Camera*)malloc(sizeof(Camera));
    
    // Extract camera position and rotation from transform matrix
    for (int i = 0; i < 4; i++) {
        json_object* row = json_object_array_get_idx(transform_matrix_obj, i);
        for (int j = 0; j < 4; j++) {
            json_object* val = json_object_array_get_idx(row, j);
            double v = json_object_get_double(val);
            
            if (j < 3 && i < 3) {
                cam->rotation[i * 3 + j] = (float)v;
            } else if (j == 3 && i < 3) {
                cam->position[i] = (float)v;
            }
        }
    }
    
    // Assume image size (you might want to load this from the actual images)
    cam->width = 400;  // Adjust based on your images
    cam->height = 400;
    
    // Calculate focal length from field of view
    cam->focal = cam->width / (2.0f * tanf(camera_angle_x / 2.0f));
    
    free(data);
    json_object_put(root);
    return cam;
}

void save_nerf(NeRF* nerf, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
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

NeRF* load_nerf(const char* filename, int custom_batch_size, int max_rays, cublasHandle_t cublas_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int pos_encode_levels, dir_encode_levels, input_dim, output_dim, stored_batch_size;
    fread(&pos_encode_levels, sizeof(int), 1, file);
    fread(&dir_encode_levels, sizeof(int), 1, file);
    fread(&input_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    fclose(file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Calculate hidden_dim from the saved MLP
    int hidden_dim = 256; // Default, you might want to save/load this too
    
    NeRF* nerf = init_nerf(pos_encode_levels, dir_encode_levels, hidden_dim, 
                           batch_size, max_rays, cublas_handle);
    
    // Load the MLP
    char mlp_filename[256];
    snprintf(mlp_filename, sizeof(mlp_filename), "%s_coarse.bin", filename);
    free_mlp(nerf->coarse_mlp);
    nerf->coarse_mlp = load_mlp(mlp_filename, batch_size, cublas_handle);
    
    printf("NeRF loaded from %s\n", filename);
    return nerf;
}