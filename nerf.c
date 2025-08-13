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

__global__ void sample_points_kernel(float* rays_o, float* rays_d, float* positions,
                                    float* z_vals, int num_rays) {
    int ray_idx = blockIdx.x;
    int sample_idx = threadIdx.x;
    
    if (ray_idx < num_rays && sample_idx < NUM_SAMPLES) {
        float t = NEAR_PLANE + (FAR_PLANE - NEAR_PLANE) * sample_idx / (NUM_SAMPLES - 1);
        int global_idx = ray_idx * NUM_SAMPLES + sample_idx;
        
        z_vals[global_idx] = t;
        
        // Just store raw 3D coordinates (no positional encoding)
        positions[global_idx * 3 + 0] = rays_o[ray_idx*3 + 0] + t * rays_d[ray_idx*3 + 0];
        positions[global_idx * 3 + 1] = rays_o[ray_idx*3 + 1] + t * rays_d[ray_idx*3 + 1];
        positions[global_idx * 3 + 2] = rays_o[ray_idx*3 + 2] + t * rays_d[ray_idx*3 + 2];
    }
}

__global__ void volume_render_kernel(float* mlp_output, float* z_vals,
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
            
            // Extract density (with ReLU) and colors (with sigmoid)
            float density = fmaxf(0.0f, mlp_output[sample_idx * 4 + 3]);
            float r = 1.0f / (1.0f + expf(-mlp_output[sample_idx * 4 + 0]));
            float g = 1.0f / (1.0f + expf(-mlp_output[sample_idx * 4 + 1]));
            float b = 1.0f / (1.0f + expf(-mlp_output[sample_idx * 4 + 2]));
            
            float alpha = 1.0f - expf(-density * delta);
            float transmittance = expf(-accumulated_alpha);
            
            accumulated_rgb[0] += transmittance * alpha * r;
            accumulated_rgb[1] += transmittance * alpha * g;
            accumulated_rgb[2] += transmittance * alpha * b;
            
            accumulated_alpha += density * delta;
        }
        
        rendered_rgb[ray_idx * 3 + 0] = accumulated_rgb[0];
        rendered_rgb[ray_idx * 3 + 1] = accumulated_rgb[1];
        rendered_rgb[ray_idx * 3 + 2] = accumulated_rgb[2];
    }
}

__global__ void calculate_loss_kernel(float* rendered_rgb, float* target_rgb,
                                     float* loss, int num_rays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rays * 3) {
        float diff = rendered_rgb[idx] - target_rgb[idx];
        loss[idx] = diff * diff;
    }
}

__global__ void volume_gradient_kernel(float* rendered_rgb, float* target_rgb,
                                      float* mlp_output, float* z_vals,
                                      float* mlp_grad, int num_rays) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx < num_rays) {
        // Gradient of loss w.r.t. rendered color
        float grad_rgb[3];
        grad_rgb[0] = 2.0f * (rendered_rgb[ray_idx * 3 + 0] - target_rgb[ray_idx * 3 + 0]);
        grad_rgb[1] = 2.0f * (rendered_rgb[ray_idx * 3 + 1] - target_rgb[ray_idx * 3 + 1]);
        grad_rgb[2] = 2.0f * (rendered_rgb[ray_idx * 3 + 2] - target_rgb[ray_idx * 3 + 2]);
        
        // Backpropagate through volume rendering
        for (int s = 0; s < NUM_SAMPLES; s++) {
            int sample_idx = ray_idx * NUM_SAMPLES + s;
            
            float delta = (s == NUM_SAMPLES - 1) ? 
                (FAR_PLANE - z_vals[sample_idx]) : 
                (z_vals[sample_idx + 1] - z_vals[sample_idx]);
            
            // Compute transmittance up to this sample
            float accumulated_alpha = 0.0f;
            for (int prev = 0; prev < s; prev++) {
                int prev_idx = ray_idx * NUM_SAMPLES + prev;
                float prev_density = fmaxf(0.0f, mlp_output[prev_idx * 4 + 3]);
                accumulated_alpha += prev_density * delta;
            }
            float transmittance = expf(-accumulated_alpha);
            
            float density = fmaxf(0.0f, mlp_output[sample_idx * 4 + 3]);
            float alpha = 1.0f - expf(-density * delta);
            
            // Sigmoid activations and their derivatives
            float r = 1.0f / (1.0f + expf(-mlp_output[sample_idx * 4 + 0]));
            float g = 1.0f / (1.0f + expf(-mlp_output[sample_idx * 4 + 1]));
            float b = 1.0f / (1.0f + expf(-mlp_output[sample_idx * 4 + 2]));
            
            // Gradients w.r.t. MLP outputs (before activation)
            mlp_grad[sample_idx * 4 + 0] = grad_rgb[0] * transmittance * alpha * r * (1.0f - r);
            mlp_grad[sample_idx * 4 + 1] = grad_rgb[1] * transmittance * alpha * g * (1.0f - g);
            mlp_grad[sample_idx * 4 + 2] = grad_rgb[2] * transmittance * alpha * b * (1.0f - b);
            
            // Gradient w.r.t. density (more complex due to transmittance effects)
            float color_contrib = grad_rgb[0] * r + grad_rgb[1] * g + grad_rgb[2] * b;
            mlp_grad[sample_idx * 4 + 3] = color_contrib * transmittance * delta * expf(-density * delta);
        }
    }
}

NeRF* init_nerf(int num_rays, cublasHandle_t cublas_handle) {
    NeRF* nerf = (NeRF*)malloc(sizeof(NeRF));
    nerf->num_rays = num_rays;
    nerf->num_samples = num_rays * NUM_SAMPLES;
    nerf->cublas_handle = cublas_handle;
    
    // Initialize MLP with simple 3D input (no positional encoding)
    const int input_dim = 3;
    const int hidden_dim = 256;
    const int output_dim = 4;
    const int batch_size = nerf->num_samples;  // Handle all samples at once
    
    nerf->position_mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size, cublas_handle);
    
    // Allocate GPU memory
    CHECK_CUDA(cudaMalloc(&nerf->d_positions, nerf->num_samples * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rendered_rgb, num_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_target_rgb, num_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rays_o, num_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_rays_d, num_rays * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_z_vals, nerf->num_samples * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_mlp_output, nerf->num_samples * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nerf->d_color_grad, nerf->num_samples * 4 * sizeof(float)));
    
    return nerf;
}

void free_nerf(NeRF* nerf) {
    free_mlp(nerf->position_mlp);
    cudaFree(nerf->d_positions);
    cudaFree(nerf->d_rendered_rgb);
    cudaFree(nerf->d_target_rgb);
    cudaFree(nerf->d_rays_o);
    cudaFree(nerf->d_rays_d);
    cudaFree(nerf->d_z_vals);
    cudaFree(nerf->d_mlp_output);
    cudaFree(nerf->d_color_grad);
    free(nerf);
}

void forward_pass(NeRF* nerf) {
    // 1. Sample 3D points along rays (no positional encoding)
    dim3 grid_dim(nerf->num_rays);
    dim3 block_dim(NUM_SAMPLES);
    sample_points_kernel<<<grid_dim, block_dim>>>(
        nerf->d_rays_o, nerf->d_rays_d, nerf->d_positions, nerf->d_z_vals, nerf->num_rays);
    
    // 2. Run MLP on all samples at once (no batching loop!)
    forward_pass_mlp(nerf->position_mlp, nerf->d_positions);
    
    // 3. Copy MLP output for volume rendering
    CHECK_CUDA(cudaMemcpy(nerf->d_mlp_output, nerf->position_mlp->d_layer2_output, 
                         nerf->num_samples * 4 * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // 4. Volume render to get final pixel colors
    int block_size = 256;
    int num_blocks = (nerf->num_rays + block_size - 1) / block_size;
    volume_render_kernel<<<num_blocks, block_size>>>(
        nerf->d_mlp_output, nerf->d_z_vals, nerf->d_rendered_rgb, nerf->num_rays);
}

float calculate_loss(NeRF* nerf) {
    int total_elements = nerf->num_rays * 3;
    
    // Allocate temporary memory for loss values
    float* d_losses;
    CHECK_CUDA(cudaMalloc(&d_losses, total_elements * sizeof(float)));
    
    // Calculate squared differences
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    calculate_loss_kernel<<<num_blocks, block_size>>>(
        nerf->d_rendered_rgb, nerf->d_target_rgb, d_losses, nerf->num_rays);
    
    // Sum up losses on GPU
    float total_loss = 0.0f;
    CHECK_CUBLAS(cublasSasum(nerf->cublas_handle, total_elements, d_losses, 1, &total_loss));
    
    CHECK_CUDA(cudaFree(d_losses));
    return total_loss / total_elements;
}

void backward_pass(NeRF* nerf) {
    // Calculate gradients through volume rendering (no batching loop!)
    int block_size = 256;
    int num_blocks = (nerf->num_rays + block_size - 1) / block_size;
    volume_gradient_kernel<<<num_blocks, block_size>>>(
        nerf->d_rendered_rgb, nerf->d_target_rgb, nerf->d_mlp_output,
        nerf->d_z_vals, nerf->d_color_grad, nerf->num_rays);
    
    // Copy gradients to MLP output layer
    CHECK_CUDA(cudaMemcpy(nerf->position_mlp->d_error_output, nerf->d_color_grad,
                         nerf->num_samples * 4 * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Backpropagate through MLP
    zero_gradients_mlp(nerf->position_mlp);
    backward_pass_mlp(nerf->position_mlp, nerf->d_positions);
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