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