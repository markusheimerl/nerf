#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nerf.h"

// Simple function to create dummy RGB data
void create_dummy_image(float** image_data, int width, int height) {
    *image_data = (float*)malloc(width * height * 3 * sizeof(float));
    
    // Create a simple gradient pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            (*image_data)[idx + 0] = (float)x / width;        // Red gradient
            (*image_data)[idx + 1] = (float)y / height;       // Green gradient  
            (*image_data)[idx + 2] = 0.5f;                    // Constant blue
        }
    }
}

int main() {
    srand(time(NULL));
    
    // Initialize CUDA and cuBLAS
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
    
    // NeRF parameters
    const int pos_encode_levels = 10;
    const int dir_encode_levels = 4;
    const int hidden_dim = 256;
    const int batch_size = 1024;
    const int max_rays = 4096;
    
    // Initialize NeRF
    NeRF* nerf = init_nerf(pos_encode_levels, dir_encode_levels, hidden_dim, 
                           batch_size, max_rays, cublas_handle);
    
    printf("NeRF initialized with input dim: %d\n", nerf->input_dim);
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 5e-4f;
    const int rays_per_epoch = 2048;
    
    // Camera setup
    const int img_width = 400;
    const int img_height = 400;
    
    Camera camera = {
        .position = {4.0f, 0.0f, 0.0f},
        .rotation = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        .focal = 200.0f,
        .width = img_width,
        .height = img_height
    };
    
    // Device memory for camera parameters
    float *d_cam_pos, *d_cam_rot;
    CHECK_CUDA(cudaMalloc(&d_cam_pos, 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cam_rot, 9 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_cam_pos, camera.position, 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cam_rot, camera.rotation, 9 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create target image
    float* image_data;
    create_dummy_image(&image_data, img_width, img_height);
    
    // Host memory for training
    int* pixel_coords = (int*)malloc(rays_per_epoch * 2 * sizeof(int));
    float* target_rgb = (float*)malloc(rays_per_epoch * 3 * sizeof(float));
    
    printf("Starting NeRF training...\n");
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        // Process in batches of rays
        for (int batch_start = 0; batch_start < rays_per_epoch; batch_start += max_rays) {
            int batch_end = (batch_start + max_rays < rays_per_epoch) ? 
                           batch_start + max_rays : rays_per_epoch;
            int current_rays = batch_end - batch_start;
            
            // Sample random pixels for this batch
            for (int i = 0; i < current_rays; i++) {
                pixel_coords[i * 2] = rand() % camera.width;
                pixel_coords[i * 2 + 1] = rand() % camera.height;
                
                // Get target RGB
                int pixel_idx = (pixel_coords[i * 2 + 1] * img_width + pixel_coords[i * 2]) * 3;
                target_rgb[i * 3 + 0] = image_data[pixel_idx + 0];
                target_rgb[i * 3 + 1] = image_data[pixel_idx + 1];
                target_rgb[i * 3 + 2] = image_data[pixel_idx + 2];
            }
            
            // Copy pixel coordinates and targets to device
            CHECK_CUDA(cudaMemcpy(nerf->d_pixel_coords, pixel_coords, 
                                 current_rays * 2 * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(nerf->d_target_rgb, target_rgb, 
                                 current_rays * 3 * sizeof(float), cudaMemcpyHostToDevice));
            
            // Generate rays
            int block_size = 256;
            int num_blocks = (current_rays + block_size - 1) / block_size;
            generate_rays_kernel<<<num_blocks, block_size>>>(
                nerf->d_rays_o, nerf->d_rays_d, nerf->d_pixel_coords,
                d_cam_pos, d_cam_rot, camera.focal,
                camera.width * 0.5f, camera.height * 0.5f, current_rays);
            
            // Render rays
            render_rays(nerf, nerf->d_rays_o, nerf->d_rays_d, current_rays);
            
            // Calculate loss
            float loss = calculate_loss_nerf(nerf, nerf->d_target_rgb, current_rays);
            total_loss += loss;
            num_batches++;
            
            // Simplified training: just update MLP weights using its own training
            // In practice, you'd need to implement proper gradient backpropagation
            // through the volume rendering equation
            if (num_batches % 10 == 0) {
                update_weights_mlp(nerf->coarse_mlp, learning_rate);
            }
        }
        printf("Epoch [%d/%d], Average Loss: %.6f\n", epoch, num_epochs, total_loss / num_batches);
    }
    
    // Save the trained NeRF
    save_nerf(nerf, "trained_nerf.bin");
    
    printf("NeRF training completed!\n");
    
    // Cleanup
    free(pixel_coords);
    free(target_rgb);
    free(image_data);
    CHECK_CUDA(cudaFree(d_cam_pos));
    CHECK_CUDA(cudaFree(d_cam_rot));
    free_nerf(nerf);
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    return 0;
}