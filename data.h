#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <png.h>
#include <json-c/json.h>
#include <time.h>

#define NUM_SAMPLES 128
#define NEAR_PLANE 2.0f
#define FAR_PLANE 6.0f
#define PI 3.14159265359f

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
    Image** images;
    Camera** cameras;
    int num_images;
} Dataset;

// Function prototypes
Image* load_png(const char* filename);
void save_png(const char* filename, unsigned char* image_data, int width, int height);
void free_image(Image* img);
Camera* load_camera(const char* filename, int frame_idx);
void free_camera(Camera* cam);
Dataset* load_dataset(const char* json_path, const char* image_dir, int max_images);
void free_dataset(Dataset* dataset);
void generate_ray(Camera* cam, int u, int v, float* ray_o, float* ray_d);
void generate_random_batch(Dataset* dataset, int rays_per_batch, float* batch_X, float* batch_colors);

#endif