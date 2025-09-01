#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "nerf.h"
#include "data.h"
#include "mlp/gpu/mlp.h"

int main(int argc, char* argv[]) {
    srand(time(NULL));
    
    // Configuration parameters
    const int rays_per_batch = 16;
    const int num_samples = 128;
    const float near_plane = 2.0f;
    const float far_plane = 6.0f;
    const int pos_enc_l = 16;
    const int dir_enc_l = 8;
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
    const int intermediate_dim = 512;  // Output of first MLP
    const int output_dim = 4;  // density + RGB
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
    printf("  Required shared mem per thread: %d\n", required_shared_mem_per_thread);
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
    
    // Initialize neural networks
    MLP* mlp1;
    MLP* mlp2;
    if (argc == 3) {
        // Continue training from existing models
        printf("Loading existing models:\n");
        printf("  Model 1: %s\n", argv[1]);
        printf("  Model 2: %s\n", argv[2]);
        
        mlp1 = load_mlp(argv[1], batch_size, cublas_handle);
        mlp2 = load_mlp(argv[2], batch_size, cublas_handle);
        
        if (!mlp1 || !mlp2) {
            fprintf(stderr, "Failed to load one or both models\n");
            if (mlp1) free_mlp(mlp1);
            if (mlp2) free_mlp(mlp2);
            return -1;
        }
        
        // Verify dimensions match what we expect
        if (mlp1->input_dim != input_dim || mlp1->output_dim != intermediate_dim ||
            mlp2->input_dim != intermediate_dim || mlp2->output_dim != output_dim) {
            fprintf(stderr, "Model dimensions don't match expected architecture\n");
            fprintf(stderr, "Expected: %d -> %d -> %d -> %d -> %d\n", 
                    input_dim, hidden_dim, intermediate_dim, hidden_dim, output_dim);
            fprintf(stderr, "Loaded: %d -> %d -> %d -> %d -> %d\n",
                    mlp1->input_dim, mlp1->hidden_dim, mlp1->output_dim, 
                    mlp2->hidden_dim, mlp2->output_dim);
            free_mlp(mlp1);
            free_mlp(mlp2);
            return -1;
        }
        
        printf("Successfully loaded both models (continuing from batch %d)\n", mlp1->t);
    } else if (argc == 1) {
        // Initialize new models
        printf("Starting new training with two-layer NeRF\n");
        mlp1 = init_mlp(input_dim, hidden_dim, intermediate_dim, batch_size, cublas_handle);
        mlp2 = init_mlp(intermediate_dim, hidden_dim, output_dim, batch_size, cublas_handle);
    } else {
        fprintf(stderr, "Usage: %s [model1.bin model2.bin]\n", argv[0]);
        fprintf(stderr, "  No args: Start new training\n");
        fprintf(stderr, "  Two args: Continue training from existing models\n");
        return -1;
    }
    
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
    
    float* d_mlp_error_output = mlp2->d_error_output;
    
    // Training parameters
    const int num_batches = 2000000;
    float learning_rate = 0.0001f;
    
    printf("Starting NeRF training with %d batches...\n", num_batches);
    printf("Batch size: %d rays, %d samples per ray\n", rays_per_batch, num_samples);
    printf("Network: %d -> %d -> %d -> %d -> %d\n", input_dim, hidden_dim, intermediate_dim, hidden_dim, output_dim);
    printf("Positional encoding: pos_L=%d, dir_L=%d, total_dim=%d\n", pos_enc_l, dir_enc_l, pe_input_dim);
    
    // Training loop
    int starting_batch = mlp1->t;
    for (int batch = starting_batch; batch < starting_batch + num_batches; batch++) {
        // Learning rate decay
        if (batch % 10000 == 0) {
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
        
        // Forward pass through both MLPs
        forward_pass_mlp(mlp1, d_batch_PE_X);
        forward_pass_mlp(mlp2, mlp1->d_layer_output);
        
        // Apply activation functions and extract densities/colors from final MLP
        int block_size = 256;
        int num_blocks = (batch_size + block_size - 1) / block_size;
        activation_kernel<<<num_blocks, block_size>>>(
            mlp2->d_layer_output, d_densities, d_colors, batch_size, output_dim);
        
        // Zero gradients for both MLPs
        zero_gradients_mlp(mlp1);
        zero_gradients_mlp(mlp2);
        
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
        
        // Backward pass through both MLPs
        backward_pass_mlp(mlp2, mlp1->d_layer_output, mlp1->d_error_output);
        backward_pass_mlp(mlp1, d_batch_PE_X, NULL);
        
        // Update weights for both MLPs
        update_weights_mlp(mlp1, learning_rate);
        update_weights_mlp(mlp2, learning_rate);
        
        // Print progress and render test images
        if ((batch + 1) % 100 == 0) {
            float total_loss = 0.0f;
            CHECK_CUDA(cudaMemcpy(&total_loss, d_loss_accum, sizeof(float), cudaMemcpyDeviceToHost));
            total_loss /= rays_per_batch;
            
            printf("Batch [%d/%d], Loss: %.6f, LR: %.6f\n", 
                   batch + 1, starting_batch + num_batches, total_loss, learning_rate);
            
            if ((batch + 1) % 5000 == 0) {
                render_test_image(mlp1, mlp2, dataset, batch + 1, cublas_handle,
                                num_samples, near_plane, far_plane, pos_enc_l, dir_enc_l,
                                pe_input_dim, render_width, render_height);
            }
        }
    }
    
    // Save trained models
    char model1_filename[64], model2_filename[64];
    time_t now = time(NULL);
    strftime(model1_filename, sizeof(model1_filename), "%Y%m%d_%H%M%S_nerf_model1.bin", localtime(&now));
    strftime(model2_filename, sizeof(model2_filename), "%Y%m%d_%H%M%S_nerf_model2.bin", localtime(&now));
    save_mlp(mlp1, model1_filename);
    save_mlp(mlp2, model2_filename);
    
    printf("\nTraining completed! Models saved to: %s and %s\n", model1_filename, model2_filename);
    
    // Cleanup
    free_dataset(dataset);
    free(batch_X);
    free(batch_PE_X);
    free(batch_true_colors);
    free_mlp(mlp1);
    free_mlp(mlp2);
    
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