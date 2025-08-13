#include "nerf.h"

int main() {
    srand(time(NULL));
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    
    const int max_rays = 1024;
    const int num_train_images = 5;
    
    NeRF* nerf = init_nerf(max_rays, cublas_handle);
    printf("NeRF initialized with MLP backend\n");
    
    Image** images = (Image**)malloc(num_train_images * sizeof(Image*));
    Camera** cameras = (Camera**)malloc(num_train_images * sizeof(Camera*));
    float** image_data = (float**)malloc(num_train_images * sizeof(float*));
    
    for (int i = 0; i < num_train_images; i++) {
        char img_path[256];
        snprintf(img_path, sizeof(img_path), "./data/r_%d.png", i);
        images[i] = load_png(img_path);
        if (!images[i]) {
            printf("Failed to load %s\n", img_path);
            return -1;
        }
        
        image_data[i] = (float*)malloc(images[i]->width * images[i]->height * 3 * sizeof(float));
        image_to_float(images[i], image_data[i]);
        
        cameras[i] = load_camera("./data/transforms.json", i);
        if (!cameras[i]) {
            printf("Failed to load camera %d\n", i);
            return -1;
        }
        
        cameras[i]->width = images[i]->width;
        cameras[i]->height = images[i]->height;
        
        printf("Loaded image %d: %dx%d\n", i, images[i]->width, images[i]->height);
    }
    
    float *d_cam_pos, *d_cam_rot;
    int *d_pixel_coords;
    CHECK_CUDA(cudaMalloc(&d_cam_pos, 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cam_rot, 9 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pixel_coords, max_rays * 2 * sizeof(int)));
    
    int* h_pixel_coords = (int*)malloc(max_rays * 2 * sizeof(int));
    float* h_target_rgb = (float*)malloc(max_rays * 3 * sizeof(float));
    
    const int num_epochs = 100;
    const float learning_rate = 1e-3f;
    
    printf("Starting training with MLP backend...\n");
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        for (int img_idx = 0; img_idx < num_train_images; img_idx++) {
            Camera* cam = cameras[img_idx];
            float* target_img = image_data[img_idx];
            
            CHECK_CUDA(cudaMemcpy(d_cam_pos, cam->position, 3 * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_cam_rot, cam->rotation, 9 * sizeof(float), cudaMemcpyHostToDevice));
            
            for (int i = 0; i < max_rays; i++) {
                int u = rand() % cam->width;
                int v = rand() % cam->height;
                h_pixel_coords[i * 2] = u;
                h_pixel_coords[i * 2 + 1] = v;
                
                int pixel_idx = (v * cam->width + u) * 3;
                h_target_rgb[i * 3 + 0] = target_img[pixel_idx + 0];
                h_target_rgb[i * 3 + 1] = target_img[pixel_idx + 1];
                h_target_rgb[i * 3 + 2] = target_img[pixel_idx + 2];
            }
            
            CHECK_CUDA(cudaMemcpy(d_pixel_coords, h_pixel_coords, max_rays * 2 * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(nerf->d_target_rgb, h_target_rgb, max_rays * 3 * sizeof(float), cudaMemcpyHostToDevice));
            
            int block_size = 256;
            int num_blocks = (max_rays + block_size - 1) / block_size;
            generate_rays_kernel<<<num_blocks, block_size>>>(
                nerf->d_rays_o, nerf->d_rays_d, d_pixel_coords,
                d_cam_pos, d_cam_rot, cam->focal,
                cam->width * 0.5f, cam->height * 0.5f, max_rays);
            
            forward_pass(nerf, nerf->d_rays_o, nerf->d_rays_d, max_rays);
            
            float loss = calculate_loss(nerf, nerf->d_target_rgb, max_rays);
            total_loss += loss;
            num_batches++;
            
            backward_pass(nerf, max_rays);
            update_weights(nerf, learning_rate);
        }
        
        if (epoch % 10 == 0) {
            printf("Epoch [%d/%d], Average Loss: %.6f\n", epoch, num_epochs, total_loss / num_batches);
        }
    }
    
printf("Training completed!\n");

// Render an image from the trained MLP
printf("Rendering image from trained model...\n");

// Use the first camera for rendering
Camera* render_cam = cameras[0];
int render_width = 200;  // Smaller for faster rendering
int render_height = 200;

// Allocate host memory for the rendered image
float* h_rendered_image = (float*)malloc(render_width * render_height * 3 * sizeof(float));
unsigned char* h_image_data = (unsigned char*)malloc(render_width * render_height * 3 * sizeof(unsigned char));

// Set up camera for rendering
CHECK_CUDA(cudaMemcpy(d_cam_pos, render_cam->position, 3 * sizeof(float), cudaMemcpyHostToDevice));
CHECK_CUDA(cudaMemcpy(d_cam_rot, render_cam->rotation, 9 * sizeof(float), cudaMemcpyHostToDevice));

// Render in batches
int total_pixels = render_width * render_height;
int pixels_done = 0;

while (pixels_done < total_pixels) {
    int current_batch = (total_pixels - pixels_done > max_rays) ? max_rays : (total_pixels - pixels_done);
    
    // Generate pixel coordinates for this batch
    for (int i = 0; i < current_batch; i++) {
        int pixel_idx = pixels_done + i;
        int u = pixel_idx % render_width;
        int v = pixel_idx / render_width;
        h_pixel_coords[i * 2] = u;
        h_pixel_coords[i * 2 + 1] = v;
    }
    
    CHECK_CUDA(cudaMemcpy(d_pixel_coords, h_pixel_coords, current_batch * 2 * sizeof(int), cudaMemcpyHostToDevice));
    
    // Generate rays
    int block_size = 256;
    int num_blocks = (current_batch + block_size - 1) / block_size;
    generate_rays_kernel<<<num_blocks, block_size>>>(
        nerf->d_rays_o, nerf->d_rays_d, d_pixel_coords,
        d_cam_pos, d_cam_rot, render_cam->focal,
        render_width * 0.5f, render_height * 0.5f, current_batch);
    
    // Forward pass
    forward_pass(nerf, nerf->d_rays_o, nerf->d_rays_d, current_batch);
    
    // Copy results to host
    CHECK_CUDA(cudaMemcpy(&h_rendered_image[pixels_done * 3], nerf->d_rendered_rgb, 
                         current_batch * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    
    pixels_done += current_batch;
    printf("\rRendering progress: %d/%d pixels", pixels_done, total_pixels);
    fflush(stdout);
}
printf("\n");

// Convert to unsigned char
for (int i = 0; i < total_pixels * 3; i++) {
    float val = h_rendered_image[i];
    val = fminf(1.0f, fmaxf(0.0f, val));  // Clamp to [0,1]
    h_image_data[i] = (unsigned char)(val * 255.0f);
}

// Save as PNG
FILE* fp = fopen("rendered_image.png", "wb");
if (fp) {
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info = png_create_info_struct(png);
    
    if (!setjmp(png_jmpbuf(png))) {
        png_init_io(png, fp);
        png_set_IHDR(png, info, render_width, render_height, 8, PNG_COLOR_TYPE_RGB,
                     PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png, info);
        
        png_bytep* row_pointers = (png_bytep*)malloc(render_height * sizeof(png_bytep));
        for (int y = 0; y < render_height; y++) {
            row_pointers[y] = h_image_data + y * render_width * 3;
        }
        
        png_write_image(png, row_pointers);
        png_write_end(png, info);
        free(row_pointers);
        printf("Image saved as rendered_image.png\n");
    }
    
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

free(h_rendered_image);
free(h_image_data);

    for (int i = 0; i < num_train_images; i++) {
        free_image(images[i]);
        free(cameras[i]);
        free(image_data[i]);
    }
    free(images);
    free(cameras);
    free(image_data);
    free(h_pixel_coords);
    free(h_target_rgb);
    CHECK_CUDA(cudaFree(d_cam_pos));
    CHECK_CUDA(cudaFree(d_cam_rot));
    CHECK_CUDA(cudaFree(d_pixel_coords));
    free_nerf(nerf);
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    return 0;
}