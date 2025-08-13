#ifndef NERF_H
#define NERF_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mlp/gpu/mlp.h"
#include "mlp/data.h"
#include "image_utils.h"

#define MAX_ENCODING_FUNCS 10
#define NUM_SAMPLES 64
#define NUM_FINE_SAMPLES 128
#define NEAR_PLANE 2.0f
#define FAR_PLANE 6.0f
#define PI 3.14159265359f

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    float position[3];  // x, y, z
    float rotation[9];  // 3x3 rotation matrix (row-major)
    float focal;        // focal length
    int width, height;  // image dimensions
} Camera;

typedef struct {
    float origin[3];     // ray origin
    float direction[3];  // ray direction (normalized)
} Ray;

typedef struct {
    MLP* coarse_mlp;    // Coarse network
    MLP* fine_mlp;      // Fine network (optional, can be NULL)
    int pos_encode_levels;  // Number of positional encoding levels
    int dir_encode_levels;  // Number of direction encoding levels
    int input_dim;      // Total input dimension after encoding
    int output_dim;     // 4 (RGB + density)
    int batch_size;
    
    // Device memory for ray processing
    float* d_rays_o;    // Ray origins
    float* d_rays_d;    // Ray directions
    float* d_points;    // Sample points along rays
    float* d_directions; // Directions at sample points
    float* d_z_vals;    // Depth values
    float* d_encoded_input; // Encoded positions and directions
    float* d_colors;    // RGB colors from network
    float* d_densities; // Volume densities
    float* d_rendered_rgb; // Final rendered colors
    float* d_target_rgb;   // Target RGB values for training
    float* d_color_grad;   // Gradients w.r.t colors
    float* d_density_grad; // Gradients w.r.t densities
    float* d_rgb_grad;     // Gradients w.r.t rendered RGB
    int* d_pixel_coords;   // Pixel coordinates
    
    cublasHandle_t cublas_handle;
} NeRF;

// CUDA kernel prototypes
__global__ void generate_rays_kernel(float* rays_o, float* rays_d, int* pixel_coords, 
                                    float* cam_pos, float* cam_rot, float focal, 
                                    float half_width, float half_height, int num_rays);
__global__ void sample_points_kernel(float* rays_o, float* rays_d, float* points, 
                                    float* directions, float* z_vals, int num_rays);
__global__ void positional_encoding_kernel(float* input, float* output, int input_dim, 
                                          int levels, int num_samples);
__global__ void volume_render_kernel(float* colors, float* densities, float* z_vals, 
                                    float* rgb_output, int num_rays);
__global__ void volume_render_backward_kernel(float* colors, float* densities, float* z_vals,
                                             float* rgb_grad, float* color_grad, float* density_grad,
                                             int num_rays);
__global__ void sigmoid_activation_kernel(float* data, int size);
__global__ void relu_activation_kernel(float* data, int size);
__global__ void mse_loss_kernel(float* rendered, float* target, float* loss, int size);
__global__ void mse_grad_kernel(float* rendered, float* target, float* grad, int size);

// Function prototypes
NeRF* init_nerf(int pos_encode_levels, int dir_encode_levels, int hidden_dim, 
                int batch_size, int max_rays, cublasHandle_t cublas_handle);
void free_nerf(NeRF* nerf);
void render_rays(NeRF* nerf, float* d_rays_o, float* d_rays_d, int num_rays);
float calculate_loss_nerf(NeRF* nerf, float* d_target_rgb, int num_rays);
void backward_pass_nerf(NeRF* nerf, float* d_target_rgb, int num_rays);
void update_weights_nerf(NeRF* nerf, float learning_rate);
Camera* load_camera_from_transforms(const char* filename, int frame_idx);
void save_nerf(NeRF* nerf, const char* filename);
NeRF* load_nerf(const char* filename, int custom_batch_size, int max_rays, cublasHandle_t cublas_handle);

#endif