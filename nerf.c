#include "nerf.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

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
        
        int out_idx = global_idx * 63;
        
        encoded_pos[out_idx++] = x;
        for (int l = 0; l < 10; l++) {
            float freq = powf(2.0f, l);
            encoded_pos[out_idx++] = sinf(freq * PI * x);
            encoded_pos[out_idx++] = cosf(freq * PI * x);
        }
        
        encoded_pos[out_idx++] = y;
        for (int l = 0; l < 10; l++) {
            float freq = powf(2.0f, l);
            encoded_pos[out_idx++] = sinf(freq * PI * y);
            encoded_pos[out_idx++] = cosf(freq * PI * y);
        }
        
        encoded_pos[out_idx++] = z;
        for (int l = 0; l < 10; l++) {
            float freq = powf(2.0f, l);
            encoded_pos[out_idx++] = sinf(freq * PI * z);
            encoded_pos[out_idx++] = cosf(freq * PI * z);
        }
    }
}

__global__ void extract_density_kernel(float* mlp_output, float* density, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        density[idx] = fmaxf(0.0f, mlp_output[idx * 4 + 3]);
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
            
            float alpha = 1.0f - expf(-density[sample_idx] * delta);
            float transmittance = expf(-accumulated_alpha);
            
            float sig_r = 1.0f / (1.0f + expf(-rgb[sample_idx * 4 + 0]));
            float sig_g = 1.0f / (1.0f + expf(-rgb[sample_idx * 4 + 1]));
            float sig_b = 1.0f / (1.0f + expf(-rgb[sample_idx * 4 + 2]));
            
            accumulated_rgb[0] += transmittance * alpha * sig_r;
            accumulated_rgb[1] += transmittance * alpha * sig_g;
            accumulated_rgb[2] += transmittance * alpha * sig_b;
            
            accumulated_alpha += density[sample_idx] * delta;
        }
        
        rendered_rgb[ray_idx * 3 + 0] = accumulated_rgb[0];
        rendered_rgb[ray_idx * 3 + 1] = accumulated_rgb[1];
        rendered_rgb[ray_idx * 3 + 2] = accumulated_rgb[2];
    }
}

__global__ void prepare_mlp_target_kernel(float* target_rgb, float* mlp_target, int num_rays) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx < num_rays) {
        for (int s = 0; s < NUM_SAMPLES; s++) {
            int sample_idx = ray_idx * NUM_SAMPLES + s;
            int mlp_idx = sample_idx * 4;
            int rgb_idx = ray_idx * 3;
            
            mlp_target[mlp_idx + 0] = target_rgb[rgb_idx + 0];
            mlp_target[mlp_idx + 1] = target_rgb[rgb_idx + 1];
            mlp_target[mlp_idx + 2] = target_rgb[rgb_idx + 2];
            mlp_target[mlp_idx + 3] = 1.0f;
        }
    }
}

__global__ void volume_gradient_kernel(float* rendered_rgb, float* target_rgb, float* mlp_output,
                                     float* z_vals, float* density, float* volume_grad, int num_rays) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx < num_rays) {
        float grad_rgb[3];
        grad_rgb[0] = 2.0f * (rendered_rgb[ray_idx * 3 + 0] - target_rgb[ray_idx * 3 + 0]);
        grad_rgb[1] = 2.0f * (rendered_rgb[ray_idx * 3 + 1] - target_rgb[ray_idx * 3 + 1]);
        grad_rgb[2] = 2.0f * (rendered_rgb[ray_idx * 3 + 2] - target_rgb[ray_idx * 3 + 2]);
        
        for (int s = 0; s < NUM_SAMPLES; s++) {
            int sample_idx = ray_idx * NUM_SAMPLES + s;
            int mlp_idx = sample_idx * 4;
            
            float delta = (s == NUM_SAMPLES - 1) ? 
                (FAR_PLANE - z_vals[sample_idx]) : 
                (z_vals[sample_idx + 1] - z_vals[sample_idx]);
            
            float accumulated_alpha = 0.0f;
            for (int prev = 0; prev < s; prev++) {
                accumulated_alpha += density[ray_idx * NUM_SAMPLES + prev] * delta;
            }
            
            float transmittance = expf(-accumulated_alpha);
            float alpha = 1.0f - expf(-density[sample_idx] * delta);
            
            float sig_r = 1.0f / (1.0f + expf(-mlp_output[mlp_idx + 0]));
            float sig_g = 1.0f / (1.0f + expf(-mlp_output[mlp_idx + 1]));
            float sig_b = 1.0f / (1.0f + expf(-mlp_output[mlp_idx + 2]));
            
            volume_grad[mlp_idx + 0] = grad_rgb[0] * transmittance * alpha * sig_r * (1.0f - sig_r);
            volume_grad[mlp_idx + 1] = grad_rgb[1] * transmittance * alpha * sig_g * (1.0f - sig_g);
            volume_grad[mlp_idx + 2] = grad_rgb[2] * transmittance * alpha * sig_b * (1.0f - sig_b);
            volume_grad[mlp_idx + 3] = (grad_rgb[0] * sig_r + grad_rgb[1] * sig_g + grad_rgb[2] * sig_b) * 
                                     transmittance * delta * expf(-density[sample_idx] * delta);
        }
    }
}

NeRF* init_nerf(int max_rays, cublasHandle_t cublas_handle) {
    NeRF* nerf = (NeRF*)malloc(sizeof(NeRF));
    nerf->max_rays = max_rays;
    nerf->max_samples = max_rays * NUM_SAMPLES;
    nerf->cublas_handle = cublas_handle;
    
    const int input_dim = 63;
    const int hidden_dim = 256;
    const int output_dim = 4;
    const int batch_size = 4096;
    
    nerf->position_mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size, cublas_handle);
    
    CHECK_CUDA(cudaMalloc(&nerf->d_encoded_pos, nerf->max_samples * 63 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_density, nerf->max_samples * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rendered_rgb, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_target_rgb, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rays_o, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rays_d, max_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_z_vals, nerf->max_samples * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_mlp_input, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_mlp_output, nerf->max_samples * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_mlp_target, nerf->max_samples * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_volume_grad, nerf->max_samples * output_dim * sizeof(float)));
    
    return nerf;
}

void free_nerf(NeRF* nerf) {
    free_mlp(nerf->position_mlp);
    cudaFree(nerf->d_encoded_pos);
    cudaFree(nerf->d_density);
    cudaFree(nerf->d_rendered_rgb);
    cudaFree(nerf->d_target_rgb);
    cudaFree(nerf->d_rays_o);
    cudaFree(nerf->d_rays_d);
    cudaFree(nerf->d_z_vals);
    cudaFree(nerf->d_mlp_input);
    cudaFree(nerf->d_mlp_output);
    cudaFree(nerf->d_mlp_target);
    cudaFree(nerf->d_volume_grad);
    free(nerf);
}

void forward_pass(NeRF* nerf, float* d_rays_o, float* d_rays_d, int num_rays) {
    dim3 grid_dim(num_rays);
    dim3 block_dim(NUM_SAMPLES);
    sample_and_encode_kernel<<<grid_dim, block_dim>>>(
        d_rays_o, d_rays_d, nerf->d_encoded_pos, nerf->d_z_vals, num_rays);
    
    int total_samples = num_rays * NUM_SAMPLES;
    int batch_size = nerf->position_mlp->batch_size;
    
    for (int i = 0; i < total_samples; i += batch_size) {
        int current_batch = min(batch_size, total_samples - i);
        
        CHECK_CUDA(cudaMemcpy(nerf->d_mlp_input, 
                             &nerf->d_encoded_pos[i * 63], 
                             current_batch * 63 * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        
        if (current_batch < batch_size) {
            CHECK_CUDA(cudaMemset(&nerf->d_mlp_input[current_batch * 63], 0, 
                                 (batch_size - current_batch) * 63 * sizeof(float)));
        }
        
        forward_pass_mlp(nerf->position_mlp, nerf->d_mlp_input);
        
        CHECK_CUDA(cudaMemcpy(&nerf->d_mlp_output[i * 4], 
                             nerf->position_mlp->d_layer2_output, 
                             current_batch * 4 * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
    }
    
    int block_size = 256;
    int num_blocks = (total_samples + block_size - 1) / block_size;
    extract_density_kernel<<<num_blocks, block_size>>>(
        nerf->d_mlp_output, nerf->d_density, total_samples);
    
    num_blocks = (num_rays + block_size - 1) / block_size;
    volume_render_kernel<<<num_blocks, block_size>>>(
        nerf->d_mlp_output, nerf->d_density, nerf->d_z_vals, 
        nerf->d_rendered_rgb, num_rays);
}

float calculate_loss(NeRF* nerf, float* d_target_rgb, int num_rays) {
    CHECK_CUDA(cudaMemcpy(nerf->d_target_rgb, d_target_rgb, 
                         num_rays * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    
    int block_size = 256;
    int num_blocks = (num_rays + block_size - 1) / block_size;
    prepare_mlp_target_kernel<<<num_blocks, block_size>>>(
        nerf->d_target_rgb, nerf->d_mlp_target, num_rays);
    
    return calculate_loss_mlp(nerf->position_mlp, nerf->d_mlp_target);
}

void backward_pass(NeRF* nerf, int num_rays) {
    zero_gradients_mlp(nerf->position_mlp);
    
    int block_size = 256;
    int num_blocks = (num_rays + block_size - 1) / block_size;
    volume_gradient_kernel<<<num_blocks, block_size>>>(
        nerf->d_rendered_rgb, nerf->d_target_rgb, nerf->d_mlp_output,
        nerf->d_z_vals, nerf->d_density, nerf->d_volume_grad, num_rays);
    
    int total_samples = num_rays * NUM_SAMPLES;
    int batch_size = nerf->position_mlp->batch_size;
    
    for (int i = 0; i < total_samples; i += batch_size) {
        int current_batch = min(batch_size, total_samples - i);
        
        CHECK_CUDA(cudaMemcpy(nerf->position_mlp->d_layer2_output, 
                             &nerf->d_volume_grad[i * 4], 
                             current_batch * 4 * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        
        CHECK_CUDA(cudaMemcpy(nerf->d_mlp_input, 
                             &nerf->d_encoded_pos[i * 63], 
                             current_batch * 63 * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        
        if (current_batch < batch_size) {
            CHECK_CUDA(cudaMemset(&nerf->d_mlp_input[current_batch * 63], 0, 
                                 (batch_size - current_batch) * 63 * sizeof(float)));
            CHECK_CUDA(cudaMemset(&nerf->position_mlp->d_layer2_output[current_batch * 4], 0, 
                                 (batch_size - current_batch) * 4 * sizeof(float)));
        }
        
        backward_pass_mlp(nerf->position_mlp, nerf->d_mlp_input);
    }
}

void update_weights(NeRF* nerf, float learning_rate) {
    update_weights_mlp(nerf->position_mlp, learning_rate);
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