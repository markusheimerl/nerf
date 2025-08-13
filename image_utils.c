#include "image_utils.h"
#include <png.h>

Image* load_png(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Could not open file %s\n", filename);
        return NULL;
    }
    
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return NULL;
    }
    
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        return NULL;
    }
    
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return NULL;
    }
    
    png_init_io(png, fp);
    png_read_info(png, info);
    
    Image* img = (Image*)malloc(sizeof(Image));
    img->width = png_get_image_width(png, info);
    img->height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);
    
    // Convert to RGB if needed
    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    
    png_read_update_info(png, info);
    
    img->channels = 4; // RGBA
    img->data = (unsigned char*)malloc(img->height * png_get_rowbytes(png, info));
    
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * img->height);
    for (int y = 0; y < img->height; y++) {
        row_pointers[y] = img->data + y * png_get_rowbytes(png, info);
    }
    
    png_read_image(png, row_pointers);
    png_read_end(png, info);
    
    free(row_pointers);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    
    return img;
}

void free_image(Image* img) {
    if (img) {
        if (img->data) free(img->data);
        free(img);
    }
}

void image_to_float(Image* img, float* output) {
    for (int i = 0; i < img->width * img->height; i++) {
        output[i * 3 + 0] = img->data[i * 4 + 0] / 255.0f; // R
        output[i * 3 + 1] = img->data[i * 4 + 1] / 255.0f; // G  
        output[i * 3 + 2] = img->data[i * 4 + 2] / 255.0f; // B
        // Ignore alpha channel
    }
}