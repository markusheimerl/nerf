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
    
    float* d_encoded_pos;
    float* d_density;
    float* d_rendered_rgb;
    float* d_target_rgb;
    float* d_rays_o;
    float* d_rays_d;
    float* d_z_vals;
    float* d_mlp_input;
    float* d_mlp_output;
    float* d_mlp_target;
    float* d_volume_grad;
    
    cublasHandle_t cublas_handle;
    int max_rays;
    int max_samples;
} NeRF;

__global__ void generate_rays_kernel(float* rays_o, float* rays_d, int* pixel_coords,
                                    float* cam_pos, float* cam_rot, float focal,
                                    float half_width, float half_height, int num_rays);
__global__ void sample_and_encode_kernel(float* rays_o, float* rays_d, float* encoded_pos,
                                        float* z_vals, int num_rays);
__global__ void extract_density_kernel(float* mlp_output, float* density, int num_samples);
__global__ void volume_render_kernel(float* rgb, float* density, float* z_vals,
                                    float* rendered_rgb, int num_rays);
__global__ void prepare_mlp_target_kernel(float* target_rgb, float* mlp_target, int num_rays);
__global__ void volume_gradient_kernel(float* rendered_rgb, float* target_rgb, float* mlp_output,
                                     float* z_vals, float* density, float* volume_grad, int num_rays);

Image* load_png(const char* filename);
void free_image(Image* img);
void image_to_float(Image* img, float* output);
NeRF* init_nerf(int max_rays, cublasHandle_t cublas_handle);
void free_nerf(NeRF* nerf);
void forward_pass(NeRF* nerf, float* d_rays_o, float* d_rays_d, int num_rays);
float calculate_loss(NeRF* nerf, float* d_target_rgb, int num_rays);
void backward_pass(NeRF* nerf, int num_rays);
void update_weights(NeRF* nerf, float learning_rate);
Camera* load_camera(const char* filename, int frame_idx);

#endif