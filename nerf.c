#include "nerf.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

// Image loading functions
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

void image_to_float(Image* img, float* output) {
    for (int i = 0; i < img->width * img->height; i++) {
        output[i * 3 + 0] = img->data[i * 4 + 0] / 255.0f;
        output[i * 3 + 1] = img->data[i * 4 + 1] / 255.0f;
        output[i * 3 + 2] = img->data[i * 4 + 2] / 255.0f;
    }
}

// CUDA kernels
__global__ void generate_rays_kernel(float* rays_o, float* rays_d, int* pixel_coords,
                                    float* cam_pos, float* cam_rot, float focal,
                                    float half_width, float half_height, int num_rays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rays) {
        int u = pixel_coords[idx * 2];
        int v = pixel_coords[idx * 2 + 1];
        
        float x = (u - half_width) / focal;
        float y = -(v - half_height) / focal;
        float z = -1.0f;
        
        float norm = sqrtf(x*x + y*y + z*z);
        x /= norm; y /= norm; z /= norm;
        
        rays_d[idx*3 + 0] = cam_rot[0]*x + cam_rot[1]*y + cam_rot[2]*z;
        rays_d[idx*3 + 1] = cam_rot[3]*x + cam_rot[4]*y + cam_rot[5]*z;
        rays_d[idx*3 + 2] = cam_rot[6]*x + cam_rot[7]*y + cam_rot[8]*z;
        
        rays_o[idx*3 + 0] = cam_pos[0];
        rays_o[idx*3 + 1] = cam_pos[1];
        rays_o[idx*3 + 2] = cam_pos[2];
    }
}

__global__ void sample_and_encode_kernel(float* rays_o, float* rays_d, float* encoded_pos,
                                        float* z_vals, int num_rays) {
    int ray_idx = blockIdx.x;
    int sample_idx = threadIdx.x;
    
    if (ray_idx < num_rays && sample_idx < NUM_SAMPLES) {
        float t = NEAR_PLANE + (FAR_PLANE - NEAR_PLANE) * sample_idx / (NUM_SAMPLES - 1);
        int global_idx = ray_idx * NUM_SAMPLES + sample_idx;
        
        z_vals[global_idx] = t;
        
        float x = rays_o[ray_idx*3 + 0] + t * rays_d[ray_idx*3 + 0];
        float y = rays_o[ray_idx*3 + 1] + t * rays_d[ray_idx*3 + 1];
        float z = rays_o[ray_idx*3 + 2] + t * rays_d[ray_idx*3 + 2];
        
        // Positional encoding (10 levels for positions)
        int out_idx = global_idx * 63; // 3 * (1 + 2*10)
        
        // x coordinate
        encoded_pos[out_idx++] = x;
        for (int l = 0; l < 10; l++) {
            float freq = powf(2.0f, l);
            encoded_pos[out_idx++] = sinf(freq * PI * x);
            encoded_pos[out_idx++] = cosf(freq * PI * x);
        }
        
        // y coordinate
        encoded_pos[out_idx++] = y;
        for (int l = 0; l < 10; l++) {
            float freq = powf(2.0f, l);
            encoded_pos[out_idx++] = sinf(freq * PI * y);
            encoded_pos[out_idx++] = cosf(freq * PI * y);
        }
        
        // z coordinate
        encoded_pos[out_idx++] = z;
        for (int l = 0; l < 10; l++) {
            float freq = powf(2.0f, l);
            encoded_pos[out_idx++] = sinf(freq * PI * z);
            encoded_pos[out_idx++] = cosf(freq * PI * z);
        }
    }
}

__global__ void relu_activation_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void volume_render_kernel(float* rgb, float* density, float* z_vals,
                                    float* rendered_rgb, int num_rays) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx < num_rays) {
        float accumulated_rgb[3] = {0.0f, 0.0f, 0.0f};
        float accumulated_alpha = 0.0f;
        
        for (int s = 0; s < NUM_SAMPLES; s++) {
            int sample_idx = ray_idx * NUM_SAMPLES + s;
            
            float delta = (s == NUM_SAMPLES - 1) ? 
                (FAR_PLANE - z_vals[sample_idx]) : 
                (z_vals[sample_idx + 1] - z_vals[sample_idx]);
            
            float alpha = 1.0f - expf(-fmaxf(0.0f, density[sample_idx]) * delta);
            float transmittance = expf(-accumulated_alpha);
            
            // Apply sigmoid to RGB values
            float sig_r = 1.0f / (1.0f + expf(-rgb[sample_idx * 3 + 0]));
            float sig_g = 1.0f / (1.0f + expf(-rgb[sample_idx * 3 + 1]));
            float sig_b = 1.0f / (1.0f + expf(-rgb[sample_idx * 3 + 2]));
            
            accumulated_rgb[0] += transmittance * alpha * sig_r;
            accumulated_rgb[1] += transmittance * alpha * sig_g;
            accumulated_rgb[2] += transmittance * alpha * sig_b;
            
            accumulated_alpha += fmaxf(0.0f, density[sample_idx]) * delta;
        }
        
        rendered_rgb[ray_idx * 3 + 0] = accumulated_rgb[0];
        rendered_rgb[ray_idx * 3 + 1] = accumulated_rgb[1];
        rendered_rgb[ray_idx * 3 + 2] = accumulated_rgb[2];
    }
}

__global__ void adamw_update_kernel(float* weight, float* grad, float* m, float* v,
                                  float lr, float beta1, float beta2, float eps,
                                  float weight_decay, int size, int batch_size, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float m_hat = m[idx] / (1.0f - powf(beta1, t));
        float v_hat = v[idx] / (1.0f - powf(beta2, t));
        
        weight[idx] = weight[idx] * (1.0f - lr * weight_decay) - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

NeRF* init_nerf(int batch_size, int max_rays, cublasHandle_t cublas_handle) {
    NeRF* nerf = (NeRF*)malloc(sizeof(NeRF));
    nerf->batch_size = batch_size;
    nerf->max_rays = max_rays;
    nerf->cublas_handle = cublas_handle;
    nerf->t = 0;
    
    // Allocate weights: W1 (256 x 63), W2 (3 x 256)
    CHECK_CUDA(cudaMalloc(&nerf->d_W1, 256 * 63 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_W2, 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_W1_grad, 256 * 63 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_W2_grad, 3 * 256 * sizeof(float)));
    
    // Adam state
    CHECK_CUDA(cudaMalloc(&nerf->d_W1_m, 256 * 63 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_W1_v, 256 * 63 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_W2_m, 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_W2_v, 3 * 256 * sizeof(float)));
    
    // Working memory
    int max_samples = max_rays * NUM_SAMPLES;
    CHECK_CUDA(cudaMalloc(&nerf->d_encoded_pos, max_samples * 63 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_hidden, max_samples * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rgb, max_samples * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_density, max_samples * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rendered_rgb, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_target_rgb, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_loss_grad, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rays_o, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rays_d, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_z_vals, max_samples * sizeof(float)));
    
    // Initialize weights
    float* h_W1 = (float*)malloc(256 * 63 * sizeof(float));
    float* h_W2 = (float*)malloc(3 * 256 * sizeof(float));
    
    float scale1 = sqrtf(6.0f / (63 + 256));
    float scale2 = sqrtf(6.0f / (256 + 3));
    
    for (int i = 0; i < 256 * 63; i++) {
        h_W1[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale1;
    }
    for (int i = 0; i < 3 * 256; i++) {
        h_W2[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale2;
    }
    
    CHECK_CUDA(cudaMemcpy(nerf->d_W1, h_W1, 256 * 63 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(nerf->d_W2, h_W2, 3 * 256 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Zero Adam state
    CHECK_CUDA(cudaMemset(nerf->d_W1_m, 0, 256 * 63 * sizeof(float)));
    CHECK_CUDA(cudaMemset(nerf->d_W1_v, 0, 256 * 63 * sizeof(float)));
    CHECK_CUDA(cudaMemset(nerf->d_W2_m, 0, 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMemset(nerf->d_W2_v, 0, 3 * 256 * sizeof(float)));
    
    free(h_W1);
    free(h_W2);
    
    return nerf;
}

void free_nerf(NeRF* nerf) {
    cudaFree(nerf->d_W1); cudaFree(nerf->d_W2);
    cudaFree(nerf->d_W1_grad); cudaFree(nerf->d_W2_grad);
    cudaFree(nerf->d_W1_m); cudaFree(nerf->d_W1_v);
    cudaFree(nerf->d_W2_m); cudaFree(nerf->d_W2_v);
    cudaFree(nerf->d_encoded_pos); cudaFree(nerf->d_hidden);
    cudaFree(nerf->d_rgb); cudaFree(nerf->d_density);
    cudaFree(nerf->d_rendered_rgb); cudaFree(nerf->d_target_rgb);
    cudaFree(nerf->d_loss_grad);
    cudaFree(nerf->d_rays_o); cudaFree(nerf->d_rays_d); cudaFree(nerf->d_z_vals);
    free(nerf);
}

void forward_pass(NeRF* nerf, float* d_rays_o, float* d_rays_d, int num_rays) {
    // Sample points and encode positions
    dim3 grid_dim(num_rays);
    dim3 block_dim(NUM_SAMPLES);
    sample_and_encode_kernel<<<grid_dim, block_dim>>>(
        d_rays_o, d_rays_d, nerf->d_encoded_pos, nerf->d_z_vals, num_rays);
    
    int total_samples = num_rays * NUM_SAMPLES;
    const float alpha = 1.0f, beta = 0.0f;
    
    // Process in batches
    for (int i = 0; i < total_samples; i += nerf->batch_size) {
        int current_batch = min(nerf->batch_size, total_samples - i);
        
        // Hidden layer: ReLU(encoded_pos * W1)
        CHECK_CUBLAS(cublasSgemm(nerf->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                256, current_batch, 63,
                                &alpha, nerf->d_W1, 63,
                                &nerf->d_encoded_pos[i * 63], 63,
                                &beta, &nerf->d_hidden[i * 256], 256));
        
        // ReLU activation
        int block_size = 256;
        int num_blocks = (current_batch * 256 + block_size - 1) / block_size;
        relu_activation_kernel<<<num_blocks, block_size>>>(&nerf->d_hidden[i * 256], current_batch * 256);
        
        // Output layer
        CHECK_CUBLAS(cublasSgemm(nerf->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                3, current_batch, 256,
                                &alpha, nerf->d_W2, 256,
                                &nerf->d_hidden[i * 256], 256,
                                &beta, &nerf->d_rgb[i * 3], 3));
        
        // Density (use last RGB channel as density for simplicity)
        CHECK_CUDA(cudaMemcpy(&nerf->d_density[i], &nerf->d_rgb[i * 3 + 2], 
                             current_batch * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    // Volume rendering
    int block_size = 256;
    int num_blocks = (num_rays + block_size - 1) / block_size;
    volume_render_kernel<<<num_blocks, block_size>>>(
        nerf->d_rgb, nerf->d_density, nerf->d_z_vals, nerf->d_rendered_rgb, num_rays);
}

float calculate_loss(NeRF* nerf, float* d_target_rgb, int num_rays) {
    const float beta = -1.0f;
    CHECK_CUBLAS(cublasScopy(nerf->cublas_handle, num_rays * 3, d_target_rgb, 1, nerf->d_target_rgb, 1));
    CHECK_CUBLAS(cublasSaxpy(nerf->cublas_handle, num_rays * 3, &beta, nerf->d_rendered_rgb, 1, nerf->d_target_rgb, 1));
    
    float loss;
    CHECK_CUBLAS(cublasSdot(nerf->cublas_handle, num_rays * 3, nerf->d_target_rgb, 1, nerf->d_target_rgb, 1, &loss));
    return loss / (2.0f * num_rays * 3);
}

void backward_pass(NeRF* nerf, int num_rays) {
    // Gradient of MSE loss
    const float alpha = 1.0f / (num_rays * 3);
    CHECK_CUBLAS(cublasScopy(nerf->cublas_handle, num_rays * 3, nerf->d_target_rgb, 1, nerf->d_loss_grad, 1));
    CHECK_CUBLAS(cublasSscal(nerf->cublas_handle, num_rays * 3, &alpha, nerf->d_loss_grad, 1));
    
    // Zero gradients
    CHECK_CUDA(cudaMemset(nerf->d_W1_grad, 0, 256 * 63 * sizeof(float)));
    CHECK_CUDA(cudaMemset(nerf->d_W2_grad, 0, 3 * 256 * sizeof(float)));
    
    // Simplified backward pass - accumulate gradients
    int total_samples = num_rays * NUM_SAMPLES;
    const float beta = 1.0f;
    
    for (int i = 0; i < total_samples; i += nerf->batch_size) {
        int current_batch = min(nerf->batch_size, total_samples - i);
        
        // Gradient w.r.t W2
        CHECK_CUBLAS(cublasSgemm(nerf->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                256, 3, current_batch,
                                &alpha, &nerf->d_hidden[i * 256], 256,
                                &nerf->d_loss_grad[(i/NUM_SAMPLES) * 3], 3,
                                &beta, nerf->d_W2_grad, 256));
        
        // Gradient w.r.t W1 (simplified)
        CHECK_CUBLAS(cublasSgemm(nerf->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                63, 256, current_batch,
                                &alpha, &nerf->d_encoded_pos[i * 63], 63,
                                &nerf->d_hidden[i * 256], 256,
                                &beta, nerf->d_W1_grad, 63));
    }
}

void update_weights(NeRF* nerf, float learning_rate) {
    nerf->t++;
    
    int block_size = 256;
    
    // Update W1
    int W1_size = 256 * 63;
    int num_blocks = (W1_size + block_size - 1) / block_size;
    adamw_update_kernel<<<num_blocks, block_size>>>(
        nerf->d_W1, nerf->d_W1_grad, nerf->d_W1_m, nerf->d_W1_v,
        learning_rate, 0.9f, 0.999f, 1e-8f, 0.01f, W1_size, nerf->batch_size, nerf->t);
    
    // Update W2
    int W2_size = 3 * 256;
    num_blocks = (W2_size + block_size - 1) / block_size;
    adamw_update_kernel<<<num_blocks, block_size>>>(
        nerf->d_W2, nerf->d_W2_grad, nerf->d_W2_m, nerf->d_W2_v,
        learning_rate, 0.9f, 0.999f, 1e-8f, 0.01f, W2_size, nerf->batch_size, nerf->t);
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