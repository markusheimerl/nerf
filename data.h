#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <png.h>
#include <json-c/json.h>

typedef struct {
    unsigned char* data;
    int width;
    int height;
    int channels;
} Image;

typedef struct {
    float position[3];
    float rotation[9];  // 3x3 rotation matrix stored row-major
    float focal;
    int width, height;
} Camera;

typedef struct {
    Image** images;
    Camera** cameras;
    int num_images;
} Dataset;

// Image I/O functions
Image* load_png(const char* filename);
void save_png(const char* filename, unsigned char* image_data, int width, int height);
void free_image(Image* img);

// Camera functions
Camera* load_camera(const char* json_path, int frame_idx);
void free_camera(Camera* cam);
void interpolate_cameras(const Camera* cam_a, const Camera* cam_b, float alpha, Camera* out_cam);

// Dataset functions
Dataset* load_dataset(const char* json_path, const char* image_dir, int max_images);
void free_dataset(Dataset* dataset);

// Ray generation
void generate_ray(const Camera* cam, int u, int v, float* ray_o, float* ray_d);

#endif