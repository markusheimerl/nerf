#ifndef NERF_H
#define NERF_H

#include "data.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA kernels
__global__ void activation_kernel(float* d_layer_output, float* d_densities, float* d_colors, 
                                 int batch_size, int output_dim);
__global__ void volume_rendering_and_loss_kernel(
    const float* d_densities, const float* d_colors,
    const float* d_true_colors, float* d_pixel_colors, 
    float* d_pixel_errors, float* d_loss_accum,
    int rays_per_batch, int num_samples);
__global__ void volume_rendering_gradient_kernel(
    const float* d_densities, const float* d_colors,
    const float* d_pixel_errors, float* d_mlp_error_output,
    int rays_per_batch, int num_samples);

// NeRF functions
void positional_encoding(const float* input, int input_dim, int L, float* output);
void batch_positional_encoding(const float* batch_X, int num_samples, int rays_per_batch, 
                             float* batch_pe_X, int pos_enc_l, int dir_enc_l, int pe_input_dim);
void generate_random_batch(const Dataset* dataset, int rays_per_batch, int num_samples,
                         float near_plane, float far_plane, float* batch_X, float* batch_colors);
void generate_interpolated_camera(const Dataset* dataset, Camera* out_cam);
void render_test_image(void* mlp1_ptr, void* mlp2_ptr, const Dataset* dataset, int batch_num, void* cublas_handle_ptr,
                      int num_samples, float near_plane, float far_plane, int pos_enc_l, int dir_enc_l,
                      int pe_input_dim, int render_width, int render_height);

#endif