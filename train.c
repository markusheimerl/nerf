#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nerf.h"

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
    
    // Load training images and cameras
    const int num_train_images = 10; // Use first 10 images for training
    Image** images = (Image**)malloc(num_train_images * sizeof(Image*));
    Camera** cameras = (Camera**)malloc(num_train_images * sizeof(Camera*));
    float** image_data = (float**)malloc(num_train_images * sizeof(float*));
    
    int img_width = 0, img_height = 0;
    
    for (int i = 0; i < num_train_images; i++) {
        // Load image
        char img_path[256];
        snprintf(img_path, sizeof(img_path), "./data/r_%d.png", i);
        images[i] = load_png(img_path);
        
        if (!images[i]) {
            printf("Failed to load image %s\n", img_path);
            return -1;
        }
        
        // Store image dimensions from first image
        if (i == 0) {
            img_width = images[i]->width;
            img_height = images[i]->height;
        }
        
        // Convert to float
        image_data[i] = (float*)malloc(images[i]->width * images[i]->height * 3 * sizeof(float));
        image_to_float(images[i], image_data[i]);
        
        // Load camera
        cameras[i] = load_camera_from_transforms("./data/transforms.json", i);
        if (!cameras[i]) {
            printf("Failed to load camera for frame %d\n", i);
            return -1;
        }
        
        // Update camera dimensions to match actual image size
        cameras[i]->width = img_width;
        cameras[i]->height = img_height;
        cameras[i]->focal = img_width / (2.0f * tanf(0.6911112070083618 / 2.0f)); // From transforms.json
        
        printf("Loaded image %d: %dx%d, focal: %.2f\n", i, img_width, img_height, cameras[i]->focal);
    }
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 5e-4f;
    const int rays_per_epoch = 2048;
    
    // Device memory for camera parameters
    float *d_cam_pos, *d_cam_rot;
    CHECK_CUDA(cudaMalloc(&d_cam_pos, 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cam_rot, 9 * sizeof(float)));
    
    // Host memory for training
    int* pixel_coords = (int*)malloc(rays_per_epoch * 2 * sizeof(int));
    float* target_rgb = (float*)malloc(rays_per_epoch * 3 * sizeof(float));
    
    printf("Starting NeRF training...\n");
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        // Randomly select an image for this epoch
        int img_idx = rand() % num_train_images;
        Camera* camera = cameras[img_idx];
        float* target_image = image_data[img_idx];
        
        // Update camera parameters on device
        CHECK_CUDA(cudaMemcpy(d_cam_pos, camera->position, 3 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_cam_rot, camera->rotation, 9 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Process in batches of rays
        for (int batch_start = 0; batch_start < rays_per_epoch; batch_start += max_rays) {
            int batch_end = (batch_start + max_rays < rays_per_epoch) ? 
                           batch_start + max_rays : rays_per_epoch;
            int current_rays = batch_end - batch_start;
            
            // Sample random pixels for this batch
            for (int i = 0; i < current_rays; i++) {
                int u = rand() % camera->width;
                int v = rand() % camera->height;
                pixel_coords[i * 2] = u;
                pixel_coords[i * 2 + 1] = v;
                
                // Get target RGB
                int pixel_idx = (v * img_width + u) * 3;
                target_rgb[i * 3 + 0] = target_image[pixel_idx + 0];
                target_rgb[i * 3 + 1] = target_image[pixel_idx + 1];
                target_rgb[i * 3 + 2] = target_image[pixel_idx + 2];
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
                d_cam_pos, d_cam_rot, camera->focal,
                camera->width * 0.5f, camera->height * 0.5f, current_rays);
            
            // Render rays
            render_rays(nerf, nerf->d_rays_o, nerf->d_rays_d, current_rays);
            
            // Calculate loss
            float loss = calculate_loss_nerf(nerf, nerf->d_target_rgb, current_rays);
            total_loss += loss;
            num_batches++;
            
            // Backward pass
            backward_pass_nerf(nerf, nerf->d_target_rgb, current_rays);
            
            // Update weights every few batches
            if (num_batches % 5 == 0) {
                update_weights_nerf(nerf, learning_rate);
            }
        }
        
        if (epoch % 10 == 0) {
            printf("Epoch [%d/%d], Average Loss: %.6f (Image %d)\n", 
                   epoch, num_epochs, total_loss / num_batches, img_idx);
        }
    }
    
    // Save the trained NeRF
    save_nerf(nerf, "trained_nerf.bin");
    
    printf("NeRF training completed!\n");
    
    // Cleanup
    for (int i = 0; i < num_train_images; i++) {
        free_image(images[i]);
        free(cameras[i]);
        free(image_data[i]);
    }
    free(images);
    free(cameras);
    free(image_data);
    free(pixel_coords);
    free(target_rgb);
    CHECK_CUDA(cudaFree(d_cam_pos));
    CHECK_CUDA(cudaFree(d_cam_rot));
    free_nerf(nerf);
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    return 0;
}