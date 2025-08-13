#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple structure to hold image data
typedef struct {
    unsigned char* data;
    int width;
    int height;
    int channels;
} Image;

// Function prototypes
Image* load_png(const char* filename);
void free_image(Image* img);
void image_to_float(Image* img, float* output);

#endif