#ifndef NERF_H
#define NERF_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mlp/mlp.h"
#include "mlp/data.h"

#define MAX_ENCODING_FUNCS 10
#define NUM_SAMPLES 64
#define NUM_FINE_SAMPLES 128
#define NEAR_PLANE 2.0f
#define FAR_PLANE 6.0f
#define PI 3.14159265359f

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
} NeRF;

// Function prototypes
NeRF* init_nerf(int pos_encode_levels, int dir_encode_levels, int hidden_dim, int batch_size);
void free_nerf(NeRF* nerf);
void positional_encoding(float* input, int input_dim, float* output, int levels);
void generate_rays(Camera* camera, float* rays_o, float* rays_d, int* pixel_coords, int num_rays);
void sample_points_along_rays(float* rays_o, float* rays_d, int num_rays, float* points, float* directions, float* z_vals);
void volume_render(float* colors, float* densities, float* z_vals, int num_rays, float* rgb_output);
void render_rays(NeRF* nerf, float* rays_o, float* rays_d, int num_rays, float* rgb_output);
Camera* load_camera_from_transforms(const char* filename, int frame_idx);
void save_nerf(NeRF* nerf, const char* filename);
NeRF* load_nerf(const char* filename, int custom_batch_size);

#endif