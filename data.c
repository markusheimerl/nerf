#include "data.h"

Image* load_png(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return NULL; }
    
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
    
    // Normalize to 8-bit RGBA
    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    
    png_read_update_info(png, info);
    img->channels = 4; // Always RGBA after processing
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

void save_png(const char* filename, unsigned char* image_data, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Could not create %s\n", filename);
        return;
    }
    
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return; }
    
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return;
    }
    
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }
    
    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
                PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png, info);
    
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = &image_data[y * width * 3];
    }
    
    png_write_image(png, row_pointers);
    png_write_end(png, info);
    
    free(row_pointers);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

void free_image(Image* img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

Camera* load_camera(const char* json_path, int frame_idx) {
    FILE* fp = fopen(json_path, "r");
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
    
    // Parse 4x4 transformation matrix
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
    
    // Set default dimensions and focal length
    cam->width = 400;
    cam->height = 400;
    cam->focal = cam->width / (2.0f * tanf(camera_angle_x / 2.0f));
    
    free(data);
    json_object_put(root);
    return cam;
}

void free_camera(Camera* cam) {
    free(cam);
}

void interpolate_cameras(const Camera* cam_a, const Camera* cam_b, float alpha, Camera* out_cam) {
    // Linear interpolation of position
    for (int i = 0; i < 3; i++) {
        out_cam->position[i] = (1.0f - alpha) * cam_a->position[i] + alpha * cam_b->position[i];
    }
    
    // Linear interpolation of rotation matrix followed by re-orthogonalization
    for (int i = 0; i < 9; i++) {
        out_cam->rotation[i] = (1.0f - alpha) * cam_a->rotation[i] + alpha * cam_b->rotation[i];
    }
    
    // Re-orthogonalize rotation matrix using Gram-Schmidt
    float* r = &out_cam->rotation[0]; // right vector (column 0)
    float* u = &out_cam->rotation[3]; // up vector (column 1)  
    float* f = &out_cam->rotation[6]; // forward vector (column 2)
    
    // Normalize forward vector
    float f_norm = sqrtf(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    for (int i = 0; i < 3; i++) f[i] /= f_norm;
    
    // Orthogonalize up vector
    float dot_uf = u[0]*f[0] + u[1]*f[1] + u[2]*f[2];
    for (int i = 0; i < 3; i++) u[i] -= dot_uf * f[i];
    float u_norm = sqrtf(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    for (int i = 0; i < 3; i++) u[i] /= u_norm;
    
    // Compute right as cross product of up and forward
    r[0] = u[1]*f[2] - u[2]*f[1];
    r[1] = u[2]*f[0] - u[0]*f[2];
    r[2] = u[0]*f[1] - u[1]*f[0];
    
    // Interpolate focal length
    out_cam->focal = (1.0f - alpha) * cam_a->focal + alpha * cam_b->focal;
    out_cam->width = cam_a->width;
    out_cam->height = cam_a->height;
}

Dataset* load_dataset(const char* json_path, const char* image_dir, int max_images) {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->images = (Image**)malloc(max_images * sizeof(Image*));
    dataset->cameras = (Camera**)malloc(max_images * sizeof(Camera*));
    dataset->num_images = 0;
    
    printf("Loading dataset from %s and %s...\n", json_path, image_dir);
    
    for (int i = 0; i < max_images; i++) {
        char img_path[512];
        snprintf(img_path, sizeof(img_path), "%s/r_%d.png", image_dir, i);
        
        Image* img = load_png(img_path);
        if (!img) {
            printf("Failed to load image %d: %s\n", i, img_path);
            continue;
        }
        
        Camera* cam = load_camera(json_path, i);
        if (!cam) {
            printf("Failed to load camera %d\n", i);
            free_image(img);
            continue;
        }
        
        // Update camera dimensions to match loaded image
        cam->width = img->width;
        cam->height = img->height;
        cam->focal = cam->width / (2.0f * tanf(0.6911112070083618 / 2.0f));
        
        dataset->images[dataset->num_images] = img;
        dataset->cameras[dataset->num_images] = cam;
        dataset->num_images++;
        
        if ((i + 1) % 10 == 0) {
            printf("Loaded %d/%d images...\n", i + 1, max_images);
        }
    }
    
    printf("Successfully loaded %d images\n", dataset->num_images);
    return dataset;
}

void free_dataset(Dataset* dataset) {
    if (dataset) {
        for (int i = 0; i < dataset->num_images; i++) {
            free_image(dataset->images[i]);
            free_camera(dataset->cameras[i]);
        }
        free(dataset->images);
        free(dataset->cameras);
        free(dataset);
    }
}

void generate_ray(const Camera* cam, int u, int v, float* ray_o, float* ray_d) {
    // Convert pixel coordinates to normalized camera coordinates
    float x = (u - cam->width * 0.5f) / cam->focal;
    float y = -(v - cam->height * 0.5f) / cam->focal;
    float z = -1.0f;
    
    // Normalize direction vector
    float norm = sqrtf(x*x + y*y + z*z);
    x /= norm; y /= norm; z /= norm;
    
    // Transform ray direction by camera rotation matrix
    ray_d[0] = cam->rotation[0]*x + cam->rotation[1]*y + cam->rotation[2]*z;
    ray_d[1] = cam->rotation[3]*x + cam->rotation[4]*y + cam->rotation[5]*z;
    ray_d[2] = cam->rotation[6]*x + cam->rotation[7]*y + cam->rotation[8]*z;
    
    // Ray origin is camera position
    ray_o[0] = cam->position[0];
    ray_o[1] = cam->position[1];
    ray_o[2] = cam->position[2];
}