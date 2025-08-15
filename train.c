#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "nerf.h"
#include "data.h"
#include "mlp/gpu/mlp.h"

int main() {
    srand(time(NULL));
    
    // Configuration parameters
    const int rays_per_batch = 8;
    const int num_samples = 64;
    const float near_plane = 2.0f;
    const float far_plane = 6.0f;
    const int pos_enc_l = 8;
    const int dir_enc_l = 4;
    const int input_pos_dim = 3;
    const int input_dir_dim = 3;
    const int render_width = 128;
    const int render_height = 128;
    
    // Compute derived dimensions
    const int raw_input_dim = 6;
    const int pos_enc_dim = input_pos_dim * (2 * pos_enc_l) + input_pos_dim;
    const int dir_enc_dim = input_dir_dim * (2 * dir_enc_l) + input_dir_dim;
    const int pe_input_dim = pos_enc_dim + dir_enc_dim;
    
    // Load dataset
    Dataset* dataset = load_dataset("./data/transforms.json", "./data", 100);
    if (!dataset) {
        fprintf(stderr, "Failed to load dataset\n");
        return -1;
    }
    
    // Network parameters
    const int input_dim = pe_input_dim;
    const int hidden_dim = 512;
    const int output_dim = 4;  // density + RGB
    const int num_layers = 1;
    const int batch_size = rays_per_batch * num_samples;
    
    // Initialize CUDA and cuBLAS
    CHECK_CUDA(cudaSetDevice(0));
    
    // Query GPU properties and calculate optimal block size ONCE
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Max shared memory per block: %zu bytes (%.1f KB)\n", 
           prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0f);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    
    // Calculate optimal block size based on shared memory constraints
    int max_shared_mem = prop.sharedMemPerBlock;
    int required_shared_mem_per_thread = num_samples * 3 * sizeof(float);
    int max_threads_by_shared_mem = max_shared_mem / required_shared_mem_per_thread;
    int ray_block_size = max_threads_by_shared_mem;
    ray_block_size = (ray_block_size / 32) * 32;
    if (ray_block_size < 32) ray_block_size = max_threads_by_shared_mem;
    ray_block_size = fmaxf(1, ray_block_size);
    
    // Also respect GPU's maximum threads per block limit
    ray_block_size = fminf(ray_block_size, prop.maxThreadsPerBlock);
    
    int ray_num_blocks = (rays_per_batch + ray_block_size - 1) / ray_block_size;
    int shared_mem_size = ray_block_size * num_samples * 3 * sizeof(float);
    
    // Print calculated configuration
    printf("\nOptimal kernel configuration:\n");
    printf("  Required shared mem per thread: %d bytes\n", required_shared_mem_per_thread);
    printf("  Max threads by shared mem: %d\n", max_threads_by_shared_mem);
    printf("  Chosen ray block size: %d\n", ray_block_size);
    printf("  Number of blocks: %d\n", ray_num_blocks);
    printf("  Total shared mem usage: %d bytes (%.1f KB)\n", 
           shared_mem_size, shared_mem_size / 1024.0f);
    printf("  Shared mem utilization: %.1f%%\n\n", 
           100.0f * shared_mem_size / max_shared_mem);
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
    
    // Initialize neural network
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, num_layers, batch_size, cublas_handle);
    
    // Allocate host memory
    float* batch_X = (float*)malloc(batch_size * raw_input_dim * sizeof(float));
    float* batch_PE_X = (float*)malloc(batch_size * pe_input_dim * sizeof(float));
    float* batch_true_colors = (float*)malloc(batch_size * 3 * sizeof(float));
    
    // Allocate device memory
    float* d_batch_PE_X;
    float* d_batch_true_colors;
    float* d_densities;
    float* d_colors;
    float* d_pixel_colors;
    float* d_pixel_errors;
    float* d_loss_accum;
    
    CHECK_CUDA(cudaMalloc(&d_batch_PE_X, batch_size * pe_input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_batch_true_colors, batch_size * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_densities, batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_colors, batch_size * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pixel_colors, rays_per_batch * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pixel_errors, rays_per_batch * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_loss_accum, sizeof(float)));
    
    float* d_mlp_error_output = mlp->d_error_output[mlp->num_layers - 1];
    
    // Training parameters
    const int num_batches = 2000000;
    float learning_rate = 0.001f;
    
    printf("Starting NeRF training with %d batches...\n", num_batches);
    printf("Batch size: %d rays, %d samples per ray\n", rays_per_batch, num_samples);
    printf("Network: %d -> %d -> %d (%d layers)\n", input_dim, hidden_dim, output_dim, num_layers);
    printf("Positional encoding: pos_L=%d, dir_L=%d, total_dim=%d\n", pos_enc_l, dir_enc_l, pe_input_dim);
    
    // Training loop
    for (int batch = 0; batch < num_batches; batch++) {
        // Learning rate decay
        if (batch % 1000 == 0) {
            learning_rate *= 0.99f;
        }
        
        // Generate training batch
        generate_random_batch(dataset, rays_per_batch, num_samples, near_plane, far_plane, 
                            batch_X, batch_true_colors);
        batch_positional_encoding(batch_X, num_samples, rays_per_batch, batch_PE_X, 
                                pos_enc_l, dir_enc_l, pe_input_dim);
        
        // Copy data to GPU
        CHECK_CUDA(cudaMemcpy(d_batch_PE_X, batch_PE_X, 
                             batch_size * pe_input_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_batch_true_colors, batch_true_colors, 
                             batch_size * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Forward pass
        forward_pass_mlp(mlp, d_batch_PE_X);
        
        // Apply activation functions and extract densities/colors
        int block_size = 256;
        int num_blocks = (batch_size + block_size - 1) / block_size;
        int last_layer = mlp->num_layers - 1;
        activation_kernel<<<num_blocks, block_size>>>(
            mlp->d_layer_output[last_layer], d_densities, d_colors, batch_size, output_dim);
        
        // Zero gradients
        zero_gradients_mlp(mlp);
        
        // Volume rendering and loss computation
        CHECK_CUDA(cudaMemset(d_loss_accum, 0, sizeof(float)));
        
        volume_rendering_and_loss_kernel<<<ray_num_blocks, ray_block_size>>>(
            d_densities, d_colors, d_batch_true_colors,
            d_pixel_colors, d_pixel_errors, d_loss_accum, 
            rays_per_batch, num_samples);
        
        // Compute gradients through volume rendering
        volume_rendering_gradient_kernel<<<ray_num_blocks, ray_block_size, shared_mem_size>>>(
            d_densities, d_colors, d_pixel_errors,
            d_mlp_error_output, rays_per_batch, num_samples);
        
        // Backward pass and weight update
        backward_pass_mlp(mlp, d_batch_PE_X);
        update_weights_mlp(mlp, learning_rate);
        
        // Print progress and render test images
        if ((batch + 1) % 100 == 0) {
            float total_loss = 0.0f;
            CHECK_CUDA(cudaMemcpy(&total_loss, d_loss_accum, sizeof(float), cudaMemcpyDeviceToHost));
            total_loss /= rays_per_batch;
            
            printf("Batch [%d/%d], Loss: %.6f, LR: %.6f\n", 
                   batch + 1, num_batches, total_loss, learning_rate);
            
            if ((batch + 1) % 5000 == 0) {
                render_test_image(mlp, dataset, batch + 1, cublas_handle,
                                num_samples, near_plane, far_plane, pos_enc_l, dir_enc_l,
                                pe_input_dim, render_width, render_height);
            }
        }
    }
    
    // Save trained model
    char model_filename[64];
    time_t now = time(NULL);
    strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_nerf_model.bin", localtime(&now));
    save_mlp(mlp, model_filename);
    
    printf("\nTraining completed! Model saved to: %s\n", model_filename);
    
    // Cleanup
    free_dataset(dataset);
    free(batch_X);
    free(batch_PE_X);
    free(batch_true_colors);
    free_mlp(mlp);
    
    CHECK_CUDA(cudaFree(d_batch_PE_X));
    CHECK_CUDA(cudaFree(d_batch_true_colors));
    CHECK_CUDA(cudaFree(d_densities));
    CHECK_CUDA(cudaFree(d_colors));
    CHECK_CUDA(cudaFree(d_pixel_colors));
    CHECK_CUDA(cudaFree(d_pixel_errors));
    CHECK_CUDA(cudaFree(d_loss_accum));
    
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    return 0;
}