#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <png.h>
#include <json-c/json.h>

#define NUM_SAMPLES 64
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

Image* load_png(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return NULL; }
    
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, NULL, NULL); fclose(fp); return NULL; }
    
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
    
    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    
    png_read_update_info(png, info);
    img->channels = 4;
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

Camera* load_camera(const char* filename, int frame_idx) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return NULL;
    
    fseek(fp, 0, SEEK_END);
    long length = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* data = (char*)malloc(length + 1);
    fread(data, 1, length, fp);
    data[length] = '\0';
    fclose(fp);
    
    json_object* root = json_tokener_parse(data);
    json_object* camera_angle_x_obj, *frames_obj;
    
    if (!json_object_object_get_ex(root, "camera_angle_x", &camera_angle_x_obj) ||
        !json_object_object_get_ex(root, "frames", &frames_obj)) {
        free(data);
        json_object_put(root);
        return NULL;
    }
    
    double camera_angle_x = json_object_get_double(camera_angle_x_obj);
    int num_frames = json_object_array_length(frames_obj);
    
    if (frame_idx >= num_frames) {
        free(data);
        json_object_put(root);
        return NULL;
    }
    
    json_object* frame = json_object_array_get_idx(frames_obj, frame_idx);
    json_object* transform_matrix_obj;
    
    if (!json_object_object_get_ex(frame, "transform_matrix", &transform_matrix_obj)) {
        free(data);
        json_object_put(root);
        return NULL;
    }
    
    Camera* cam = (Camera*)malloc(sizeof(Camera));
    
    for (int i = 0; i < 4; i++) {
        json_object* row = json_object_array_get_idx(transform_matrix_obj, i);
        for (int j = 0; j < 4; j++) {
            json_object* val = json_object_array_get_idx(row, j);
            double v = json_object_get_double(val);
            
            if (j < 3 && i < 3) {
                cam->rotation[i * 3 + j] = (float)v;
            } else if (j == 3 && i < 3) {
                cam->position[i] = (float)v;
            }
        }
    }
    
    cam->width = 400;
    cam->height = 400;
    cam->focal = cam->width / (2.0f * tanf(camera_angle_x / 2.0f));
    
    free(data);
    json_object_put(root);
    return cam;
}

void generate_ray(Camera* cam, int u, int v, float* ray_o, float* ray_d) {
    // Convert pixel coordinates to normalized camera coordinates
    float x = (u - cam->width * 0.5f) / cam->focal;
    float y = -(v - cam->height * 0.5f) / cam->focal;
    float z = -1.0f;
    
    // Normalize direction
    float norm = sqrtf(x*x + y*y + z*z);
    x /= norm; y /= norm; z /= norm;
    
    // Transform ray direction by camera rotation
    ray_d[0] = cam->rotation[0]*x + cam->rotation[1]*y + cam->rotation[2]*z;
    ray_d[1] = cam->rotation[3]*x + cam->rotation[4]*y + cam->rotation[5]*z;
    ray_d[2] = cam->rotation[6]*x + cam->rotation[7]*y + cam->rotation[8]*z;
    
    // Ray origin is camera position
    ray_o[0] = cam->position[0];
    ray_o[1] = cam->position[1];
    ray_o[2] = cam->position[2];
}

int main() {
    const int num_images = 100;
    const int rays_per_image = 1024;
    
    printf("Loading images and generating NeRF training data...\n");
    
    // Get timestamp for filename
    time_t now = time(NULL);
    char data_fname[64];
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    
    // Open output CSV file
    FILE* csv = fopen(data_fname, "w");
    if (!csv) {
        fprintf(stderr, "Failed to open output CSV file\n");
        return -1;
    }
    
    // Write CSV header
    fprintf(csv, "image_filename,ray_id,sample_id,x,y,z,dx,dy,dz,r_true,g_true,b_true\n");
    
    int global_ray_id = 0;
    
    // Process each image
    for (int img_idx = 0; img_idx < num_images; img_idx++) {
        printf("Processing image %d/%d...\n", img_idx + 1, num_images);
        
        // Load image
        char img_path[256];
        snprintf(img_path, sizeof(img_path), "./data/r_%d.png", img_idx);
        char img_filename[256];
        snprintf(img_filename, sizeof(img_filename), "r_%d.png", img_idx);
        
        Image* img = load_png(img_path);
        if (!img) {
            fprintf(stderr, "Failed to load %s\n", img_path);
            continue;
        }
        
        // Load camera
        Camera* cam = load_camera("./data/transforms.json", img_idx);
        if (!cam) {
            fprintf(stderr, "Failed to load camera %d\n", img_idx);
            free_image(img);
            continue;
        }
        
        cam->width = img->width;
        cam->height = img->height;
        
        // Sample random rays from this image
        for (int ray_idx = 0; ray_idx < rays_per_image; ray_idx++) {
            // Random pixel coordinates
            int u = rand() % cam->width;
            int v = rand() % cam->height;
            
            // Get pixel color (convert from RGBA to RGB and normalize)
            int pixel_idx = (v * cam->width + u) * 4;  // 4 channels (RGBA)
            float r_true = img->data[pixel_idx + 0] / 255.0f;
            float g_true = img->data[pixel_idx + 1] / 255.0f;
            float b_true = img->data[pixel_idx + 2] / 255.0f;
            
            // Generate ray
            float ray_o[3], ray_d[3];
            generate_ray(cam, u, v, ray_o, ray_d);
            
            // Sample points along the ray
            for (int sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) {
                float t = NEAR_PLANE + (FAR_PLANE - NEAR_PLANE) * sample_idx / (NUM_SAMPLES - 1);
                
                // 3D position along ray
                float x = ray_o[0] + t * ray_d[0];
                float y = ray_o[1] + t * ray_d[1];
                float z = ray_o[2] + t * ray_d[2];
                
                // Write to CSV
                fprintf(csv, "%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                       img_filename, global_ray_id, sample_idx, x, y, z, 
                       ray_d[0], ray_d[1], ray_d[2],
                       r_true, g_true, b_true);
            }
            
            global_ray_id++;
        }
        
        free_image(img);
        free(cam);
    }
    
    fclose(csv);
    
    printf("Generated training data for %d rays with %d samples each\n", 
           global_ray_id, NUM_SAMPLES);
    printf("Total data points: %d\n", global_ray_id * NUM_SAMPLES);
    printf("Data saved to: %s\n", data_fname);
    
    return 0;
}