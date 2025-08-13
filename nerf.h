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

#define NUM_SAMPLES 64
#define NEAR_PLANE 2.0f
#define FAR_PLANE 6.0f
#define PI 3.14159265359f

// CUDA Error checking macro
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
    // Network weights
    float* d_W1;      // 256 x 63 (positions encoded)
    float* d_W2;      // 3 x 256 (RGB output)
    float* d_W1_grad;
    float* d_W2_grad;
    
    // Adam optimizer state
    float* d_W1_m, *d_W1_v;
    float* d_W2_m, *d_W2_v;
    int t;
    
    // Working memory
    float* d_encoded_pos;    // batch_size x 63
    float* d_hidden;         // batch_size x 256  
    float* d_rgb;            // batch_size x 3
    float* d_density;        // batch_size x 1
    float* d_rendered_rgb;   // num_rays x 3
    float* d_target_rgb;     // num_rays x 3
    float* d_loss_grad;      // num_rays x 3
    
    // Ray data
    float* d_rays_o;         // num_rays x 3
    float* d_rays_d;         // num_rays x 3
    float* d_z_vals;         // num_rays x NUM_SAMPLES
    
    cublasHandle_t cublas_handle;
    int batch_size;
    int max_rays;
} NeRF;

// CUDA kernel prototypes
__global__ void generate_rays_kernel(float* rays_o, float* rays_d, int* pixel_coords,
                                    float* cam_pos, float* cam_rot, float focal,
                                    float half_width, float half_height, int num_rays);
__global__ void sample_and_encode_kernel(float* rays_o, float* rays_d, float* encoded_pos,
                                        float* z_vals, int num_rays);
__global__ void relu_activation_kernel(float* data, int size);
__global__ void volume_render_kernel(float* rgb, float* density, float* z_vals,
                                    float* rendered_rgb, int num_rays);
__global__ void adamw_update_kernel(float* weight, float* grad, float* m, float* v,
                                  float lr, float beta1, float beta2, float eps,
                                  float weight_decay, int size, int batch_size, int t);

// Function prototypes
Image* load_png(const char* filename);
void free_image(Image* img);
void image_to_float(Image* img, float* output);

NeRF* init_nerf(int batch_size, int max_rays, cublasHandle_t cublas_handle);
void free_nerf(NeRF* nerf);
void forward_pass(NeRF* nerf, float* d_rays_o, float* d_rays_d, int num_rays);
float calculate_loss(NeRF* nerf, float* d_target_rgb, int num_rays);
void backward_pass(NeRF* nerf, int num_rays);
void update_weights(NeRF* nerf, float learning_rate);
Camera* load_camera(const char* filename, int frame_idx);

#endif