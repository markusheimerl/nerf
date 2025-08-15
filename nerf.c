#include "nerf.h"
#include "mlp/gpu/mlp.h"
#include <math.h>
#include <time.h>

// CUDA kernel implementations
__global__ void activation_kernel(float* d_layer_output, float* d_densities, float* d_colors, 
                                 int batch_size, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int base = idx * output_dim;
        // Density: ReLU activation
        d_densities[idx] = fmaxf(0.0f, d_layer_output[base + 0]);
        // RGB: Sigmoid activation
        d_colors[idx * 3 + 0] = 1.0f / (1.0f + expf(-d_layer_output[base + 1]));
        d_colors[idx * 3 + 1] = 1.0f / (1.0f + expf(-d_layer_output[base + 2]));
        d_colors[idx * 3 + 2] = 1.0f / (1.0f + expf(-d_layer_output[base + 3]));
    }
}

__global__ void volume_rendering_and_loss_kernel(
    const float* d_densities, const float* d_colors,
    const float* d_true_colors, float* d_pixel_colors, 
    float* d_pixel_errors, float* d_loss_accum,
    int rays_per_batch, int num_samples) {
    
    int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray < rays_per_batch) {
        float pixel_color[3] = {0.0f, 0.0f, 0.0f};
        float transmittance = 1.0f;
        
        // Volume rendering integration
        for (int s = 0; s < num_samples; s++) {
            float density = d_densities[ray * num_samples + s];
            float alpha = 1.0f - expf(-density * 0.01f);
            float weight = alpha * transmittance;
            
            for (int c = 0; c < 3; c++) {
                pixel_color[c] += weight * d_colors[(ray * num_samples + s) * 3 + c];
            }
            
            transmittance *= (1.0f - alpha);
            if (transmittance < 0.01f) break; // Early termination
        }
        
        // Store rendered pixel color
        for (int c = 0; c < 3; c++) {
            d_pixel_colors[ray * 3 + c] = pixel_color[c];
        }
        
        // Compute loss and gradients
        float pixel_loss = 0.0f;
        for (int c = 0; c < 3; c++) {
            float error = pixel_color[c] - d_true_colors[ray * num_samples * 3 + c];
            d_pixel_errors[ray * 3 + c] = error;
            pixel_loss += error * error;
        }
        
        atomicAdd(d_loss_accum, pixel_loss);
    }
}

__global__ void volume_rendering_gradient_kernel(
    const float* d_densities, const float* d_colors,
    const float* d_pixel_errors, float* d_mlp_error_output,
    int rays_per_batch, int num_samples) {
    
    int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray < rays_per_batch) {
        // Dynamically allocate arrays based on num_samples
        extern __shared__ float shared_mem[];
        float* alphas = shared_mem + threadIdx.x * num_samples * 3;
        float* transmittance = alphas + num_samples;
        float* weights = transmittance + num_samples;
        
        // Precompute alpha and transmittance values
        for (int s = 0; s < num_samples; s++) {
            float density = d_densities[ray * num_samples + s];
            alphas[s] = 1.0f - expf(-density * 0.01f);
        }
        
        transmittance[0] = 1.0f;
        weights[0] = alphas[0] * transmittance[0];
        for (int s = 1; s < num_samples; s++) {
            transmittance[s] = transmittance[s - 1] * (1.0f - alphas[s - 1]);
            weights[s] = alphas[s] * transmittance[s];
        }
        
        // Compute gradients for each sample
        for (int s = 0; s < num_samples; s++) {
            // Color gradients (through sigmoid derivative)
            for (int c = 0; c < 3; c++) {
                float sigmoid_val = d_colors[(ray * num_samples + s) * 3 + c];
                float sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
                float grad = weights[s] * d_pixel_errors[ray * 3 + c] * sigmoid_deriv;
                d_mlp_error_output[(ray * num_samples + s) * 4 + 1 + c] = grad;
            }
            
            // Density gradients (more complex due to transmittance chain rule)
            float density = d_densities[ray * num_samples + s];
            float dalpha_ddensity = 0.01f * expf(-density * 0.01f);
            
            // Direct contribution from current sample
            float density_gradient = dalpha_ddensity * transmittance[s] *
                (d_pixel_errors[ray * 3 + 0] * d_colors[(ray * num_samples + s) * 3 + 0] +
                 d_pixel_errors[ray * 3 + 1] * d_colors[(ray * num_samples + s) * 3 + 1] +
                 d_pixel_errors[ray * 3 + 2] * d_colors[(ray * num_samples + s) * 3 + 2]);
            
            // Indirect contribution through transmittance of later samples
            for (int t = s + 1; t < num_samples; t++) {
                float dtrans_ddensity = -dalpha_ddensity;
                for (int k = s + 1; k <= t; k++) {
                    dtrans_ddensity *= (1.0f - alphas[k - 1]);
                }
                float dweight_t_ddensity = alphas[t] * dtrans_ddensity;
                density_gradient += dweight_t_ddensity *
                    (d_pixel_errors[ray * 3 + 0] * d_colors[(ray * num_samples + t) * 3 + 0] +
                     d_pixel_errors[ray * 3 + 1] * d_colors[(ray * num_samples + t) * 3 + 1] +
                     d_pixel_errors[ray * 3 + 2] * d_colors[(ray * num_samples + t) * 3 + 2]);
            }
            
            // Apply ReLU derivative
            float relu_deriv = density > 0.0f ? 1.0f : 0.0f;
            d_mlp_error_output[(ray * num_samples + s) * 4 + 0] = density_gradient * relu_deriv;
        }
    }
}

void positional_encoding(const float* input, int input_dim, int L, float* output) {
    int out_idx = 0;
    
    // Identity encoding
    for (int i = 0; i < input_dim; i++) {
        output[out_idx++] = input[i];
    }
    
    // Sinusoidal encoding at different frequencies
    for (int l = 0; l < L; l++) {
        float freq = powf(2.0f, (float)l);
        for (int i = 0; i < input_dim; i++) {
            output[out_idx++] = sinf(freq * M_PI * input[i]);
            output[out_idx++] = cosf(freq * M_PI * input[i]);
        }
    }
}

void batch_positional_encoding(const float* batch_X, int num_samples, int rays_per_batch, 
                             float* batch_pe_X, int pos_enc_l, int dir_enc_l, int pe_input_dim) {
    const int input_pos_dim = 3;
    const int input_dir_dim = 3;
    const int pos_enc_dim = input_pos_dim * (2 * pos_enc_l) + input_pos_dim;
    
    for (int idx = 0; idx < num_samples * rays_per_batch; idx++) {
        const float* pos = &batch_X[idx * 6 + 0];  // Position (xyz)
        const float* dir = &batch_X[idx * 6 + 3];  // Direction (xyz)
        float* out = &batch_pe_X[idx * pe_input_dim];
        
        // Encode position and direction separately
        positional_encoding(pos, input_pos_dim, pos_enc_l, out);
        positional_encoding(dir, input_dir_dim, dir_enc_l, out + pos_enc_dim);
    }
}

void generate_random_batch(const Dataset* dataset, int rays_per_batch, int num_samples,
                         float near_plane, float far_plane, float* batch_X, float* batch_colors) {
    for (int ray = 0; ray < rays_per_batch; ray++) {
        // Select random image and pixel
        int img_idx = rand() % dataset->num_images;
        Image* img = dataset->images[img_idx];
        Camera* cam = dataset->cameras[img_idx];
        
        int u = rand() % cam->width;
        int v = rand() % cam->height;
        
        // Get ground truth pixel color
        int pixel_idx = (v * cam->width + u) * 4;  // RGBA format
        float r_true = img->data[pixel_idx + 0] / 255.0f;
        float g_true = img->data[pixel_idx + 1] / 255.0f;
        float b_true = img->data[pixel_idx + 2] / 255.0f;
        
        // Generate camera ray
        float ray_o[3], ray_d[3];
        generate_ray(cam, u, v, ray_o, ray_d);
        
        // Sample points along the ray
        for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
            float t = near_plane + (far_plane - near_plane) * sample_idx / (num_samples - 1);
            int batch_idx = ray * num_samples + sample_idx;
            
            // 3D position along ray
            batch_X[batch_idx * 6 + 0] = ray_o[0] + t * ray_d[0];
            batch_X[batch_idx * 6 + 1] = ray_o[1] + t * ray_d[1];
            batch_X[batch_idx * 6 + 2] = ray_o[2] + t * ray_d[2];
            
            // Ray direction (same for all samples along this ray)
            batch_X[batch_idx * 6 + 3] = ray_d[0];
            batch_X[batch_idx * 6 + 4] = ray_d[1];
            batch_X[batch_idx * 6 + 5] = ray_d[2];
            
            // Ground truth color (same for all samples along this ray)
            batch_colors[batch_idx * 3 + 0] = r_true;
            batch_colors[batch_idx * 3 + 1] = g_true;
            batch_colors[batch_idx * 3 + 2] = b_true;
        }
    }
}

void generate_interpolated_camera(const Dataset* dataset, Camera* out_cam) {
    int cam_a_idx = rand() % dataset->num_images;
    int cam_b_idx = rand() % dataset->num_images;
    while (cam_b_idx == cam_a_idx && dataset->num_images > 1) {
        cam_b_idx = rand() % dataset->num_images;
    }
    
    float alpha = (float)rand() / (float)RAND_MAX;
    interpolate_cameras(dataset->cameras[cam_a_idx], dataset->cameras[cam_b_idx], alpha, out_cam);
}

void render_test_image(void* mlp_ptr, const Dataset* dataset, int batch_num, void* cublas_handle_ptr,
                      int num_samples, float near_plane, float far_plane, int pos_enc_l, int dir_enc_l,
                      int pe_input_dim, int render_width, int render_height) {
    MLP* mlp = (MLP*)mlp_ptr;
    cublasHandle_t cublas_handle = (cublasHandle_t)cublas_handle_ptr;
    
    printf("  Rendering test image %dx%d with interpolated view...\n", render_width, render_height);

    Camera novel_cam;
    generate_interpolated_camera(dataset, &novel_cam);

    // Create temporary MLP for rendering with single sample batch
    MLP* temp_mlp = init_mlp(mlp->input_dim, mlp->hidden_dim, mlp->output_dim, mlp->num_layers, num_samples, cublas_handle);
    
    // Copy weights from trained model
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_size = (layer == 0) ? mlp->input_dim : mlp->hidden_dim;
        int output_size = (layer == mlp->num_layers - 1) ? mlp->output_dim : mlp->hidden_dim;
        
        cudaMemcpy(temp_mlp->d_W1[layer], mlp->d_W1[layer], 
                   mlp->hidden_dim * input_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(temp_mlp->d_W2[layer], mlp->d_W2[layer], 
                   output_size * mlp->hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(temp_mlp->d_W3[layer], mlp->d_W3[layer], 
                   output_size * input_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Allocate host memory
    float* ray_X = (float*)malloc(num_samples * 6 * sizeof(float));
    float* ray_PE_X = (float*)malloc(num_samples * pe_input_dim * sizeof(float));
    unsigned char* image_data = (unsigned char*)malloc(render_width * render_height * 3);

    // Allocate device memory
    float* d_ray_PE_X;
    float* d_densities;
    float* d_colors;
    cudaMalloc(&d_ray_PE_X, num_samples * pe_input_dim * sizeof(float));
    cudaMalloc(&d_densities, num_samples * sizeof(float));
    cudaMalloc(&d_colors, num_samples * 3 * sizeof(float));

    // Render each pixel
    for (int v = 0; v < render_height; v++) {
        for (int u = 0; u < render_width; u++) {
            // Scale coordinates to match original camera resolution
            int scaled_u = (int)(u * (float)novel_cam.width / render_width);
            int scaled_v = (int)(v * (float)novel_cam.height / render_height);
            
            float ray_o[3], ray_d[3];
            generate_ray(&novel_cam, scaled_u, scaled_v, ray_o, ray_d);
            
            // Sample points along the ray
            for (int s = 0; s < num_samples; s++) {
                float t = near_plane + (far_plane - near_plane) * s / (num_samples - 1);
                ray_X[s * 6 + 0] = ray_o[0] + t * ray_d[0];
                ray_X[s * 6 + 1] = ray_o[1] + t * ray_d[1];
                ray_X[s * 6 + 2] = ray_o[2] + t * ray_d[2];
                ray_X[s * 6 + 3] = ray_d[0];
                ray_X[s * 6 + 4] = ray_d[1];
                ray_X[s * 6 + 5] = ray_d[2];
            }
            
            // Apply positional encoding and run inference
            batch_positional_encoding(ray_X, num_samples, 1, ray_PE_X, pos_enc_l, dir_enc_l, pe_input_dim);
            cudaMemcpy(d_ray_PE_X, ray_PE_X, num_samples * pe_input_dim * sizeof(float), cudaMemcpyHostToDevice);

            forward_pass_mlp(temp_mlp, d_ray_PE_X);

            // Apply activation functions and extract densities/colors directly
            int block_size = 64;
            int num_blocks = (num_samples + block_size - 1) / block_size;
            int last_layer = temp_mlp->num_layers - 1;
            activation_kernel<<<num_blocks, block_size>>>(
                temp_mlp->d_layer_output[last_layer], d_densities, d_colors, num_samples, temp_mlp->output_dim);

            // Copy results back to host
            float densities[num_samples], colors[num_samples * 3];
            cudaMemcpy(densities, d_densities, num_samples * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(colors, d_colors, num_samples * 3 * sizeof(float), cudaMemcpyDeviceToHost);

            // Perform volume rendering
            float pixel_color[3] = {0.0f, 0.0f, 0.0f};
            float transmittance = 1.0f;
            for (int s = 0; s < num_samples; s++) {
                float alpha = 1.0f - expf(-densities[s] * 0.01f);
                float weight = alpha * transmittance;
                for (int c = 0; c < 3; c++) {
                    pixel_color[c] += weight * colors[s * 3 + c];
                }
                transmittance *= (1.0f - alpha);
                if (transmittance < 0.01f) break;
            }
            
            // Store pixel in image
            int pixel_idx = (v * render_width + u) * 3;
            for (int c = 0; c < 3; c++) {
                image_data[pixel_idx + c] = (unsigned char)(
                    fminf(1.0f, fmaxf(0.0f, pixel_color[c])) * 255);
            }
        }
        
        // Progress indicator
        if ((v + 1) % (render_height / 10) == 0) {
            printf("    %d%% complete\n", (v + 1) * 100 / render_height);
        }
    }

    // Save rendered image
    char png_filename[128];
    snprintf(png_filename, sizeof(png_filename), "%06d_sample.png", batch_num);
    save_png(png_filename, image_data, render_width, render_height);
    printf("  Test image saved as %s\n\n", png_filename);

    // Cleanup
    free(image_data);
    free(ray_X);
    free(ray_PE_X);
    free_mlp(temp_mlp);
    cudaFree(d_ray_PE_X);
    cudaFree(d_densities);
    cudaFree(d_colors);
}