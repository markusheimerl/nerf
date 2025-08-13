#ifndef NERF_H
#define NERF_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <png.h>
#include <json-c/json.h>
#include "mlp/gpu/mlp.h"

#define NUM_SAMPLES 64
#define NEAR_PLANE 2.0f
#define FAR_PLANE 6.0f
#define PI 3.14159265359f

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

typedef struct {
    unsigned char* data;
    int width;
    int height;
    int channels;
} Image;

typedef struct {
    float position[3];
    float rotation[9];
    float focal;
    int width, height;
} Camera;

typedef struct {
    MLP* position_mlp;
    
    float* d_positions;      // 3D positions
    float* d_rendered_rgb;   // Final rendered colors
    float* d_target_rgb;     // Target pixel colors
    float* d_rays_o;         // Ray origins
    float* d_rays_d;         // Ray directions
    float* d_z_vals;         // Depth values
    float* d_mlp_output;     // MLP output (R,G,B,density)
    float* d_color_grad;     // Gradients w.r.t. rendered colors
    
    cublasHandle_t cublas_handle;
    int num_rays;
    int num_samples;
} NeRF;

// CUDA kernels
__global__ void sample_points_kernel(float* rays_o, float* rays_d, float* positions,
                                    float* z_vals, int num_rays);
__global__ void volume_render_kernel(float* mlp_output, float* z_vals,
                                   float* rendered_rgb, int num_rays);
__global__ void calculate_loss_kernel(float* rendered_rgb, float* target_rgb,
                                     float* loss, int num_rays);
__global__ void volume_gradient_kernel(float* rendered_rgb, float* target_rgb,
                                      float* mlp_output, float* z_vals,
                                      float* mlp_grad, int num_rays);
__global__ void generate_rays_kernel(float* rays_o, float* rays_d, int* pixel_coords,
                                    float* cam_pos, float* cam_rot, float focal,
                                    float half_width, float half_height, int num_rays);

// Function prototypes
Image* load_png(const char* filename);
void free_image(Image* img);
void image_to_float(Image* img, float* output);
NeRF* init_nerf(int num_rays, cublasHandle_t cublas_handle);
void free_nerf(NeRF* nerf);
void forward_pass(NeRF* nerf);
float calculate_loss(NeRF* nerf);
void backward_pass(NeRF* nerf);
void update_weights(NeRF* nerf, float learning_rate);
Camera* load_camera(const char* filename, int frame_idx);

#endif