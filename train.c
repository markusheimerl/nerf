#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mlp/gpu/mlp.h"
#include "data.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define RAYS_PER_BATCH 8
#define RENDER_WIDTH 128
#define RENDER_HEIGHT 128

// --- Positional Encoding Parameters ---
#define POS_ENC_L 10 // Number of frequency bands for position (typically 10 for NeRF)
#define DIR_ENC_L 4  // Number of frequency bands for direction (typically 4 for NeRF)
#define INPUT_POS_DIM 3
#define INPUT_DIR_DIM 3

#define RAW_INPUT_DIM 6
#define POS_ENC_DIM (INPUT_POS_DIM * (2 * POS_ENC_L) + INPUT_POS_DIM) // 3 + 3*2*10 = 3 + 60 = 63
#define DIR_ENC_DIM (INPUT_DIR_DIM * (2 * DIR_ENC_L) + INPUT_DIR_DIM) // 3 + 3*2*4 = 3 + 24 = 27
#define PE_INPUT_DIM (POS_ENC_DIM + DIR_ENC_DIM) // 63 + 27 = 90

// CUDA kernel for activations (output_dim=4: density, rgb)
__global__ void activation_kernel(float* d_layer2_preact, float* d_layer2_output, int batch_size, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int base = idx * output_dim;
        // Density: ReLU
        d_layer2_output[base + 0] = fmaxf(0.0f, d_layer2_preact[base + 0]);
        // RGB: Sigmoid
        for (int c = 1; c < 4; c++) {
            d_layer2_output[base + c] = 1.0f / (1.0f + expf(-d_layer2_preact[base + c]));
        }
    }
}

// CUDA kernel to extract densities and colors per sample
__global__ void extract_densities_colors_kernel(const float* d_layer2_output, float* d_densities, float* d_colors, int batch_size, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int base = idx * output_dim;
        d_densities[idx] = d_layer2_output[base + 0];
        d_colors[idx * 3 + 0] = d_layer2_output[base + 1];
        d_colors[idx * 3 + 1] = d_layer2_output[base + 2];
        d_colors[idx * 3 + 2] = d_layer2_output[base + 3];
    }
}

// ---- Positional Encoding CPU implementation ----
void positional_encoding(const float* input, int input_dim, int L, float* output) {
    // input: [x, y, z] or [dx, dy, dz]
    // output: [x, y, z, sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^{L-1} pi x), cos(2^{L-1} pi x), ...]
    int out_idx = 0;
    // Copy raw input
    for (int i = 0; i < input_dim; i++) {
        output[out_idx++] = input[i];
    }
    // Frequencies
    for (int l = 0; l < L; l++) {
        float freq = powf(2.0f, (float)l);
        for (int i = 0; i < input_dim; i++) {
            output[out_idx++] = sinf(freq * M_PI * input[i]);
            output[out_idx++] = cosf(freq * M_PI * input[i]);
        }
    }
}

// Batch positional encoding for all samples (for NUM_SAMPLES * rays)
void batch_positional_encoding(const float* batch_X, int num_samples, int rays_per_batch, float* batch_pe_X) {
    // batch_X: [NUM_SAMPLES * rays_per_batch, 6]
    // batch_pe_X: [NUM_SAMPLES * rays_per_batch, PE_INPUT_DIM]
    for (int idx = 0; idx < num_samples * rays_per_batch; idx++) {
        const float* pos = &batch_X[idx * 6 + 0];
        const float* dir = &batch_X[idx * 6 + 3];
        float* out = &batch_pe_X[idx * PE_INPUT_DIM];
        positional_encoding(pos, INPUT_POS_DIM, POS_ENC_L, out);
        positional_encoding(dir, INPUT_DIR_DIM, DIR_ENC_L, out + POS_ENC_DIM);
    }
}

// CUDA kernel for volume rendering and loss computation
__global__ void volume_rendering_and_loss_kernel(
    const float* d_densities, const float* d_colors,
    const float* d_true_colors,
    float* d_pixel_colors, float* d_pixel_errors, float* d_loss_accum,
    int rays_per_batch, int num_samples)
{
    int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray < rays_per_batch) {
        float pixel_color[3] = {0.0f, 0.0f, 0.0f};
        float transmittance = 1.0f;
        for (int s = 0; s < num_samples; s++) {
            float density = d_densities[ray * num_samples + s];
            float alpha = 1.0f - expf(-density * 0.01f);
            float weight = alpha * transmittance;
            for (int c = 0; c < 3; c++)
                pixel_color[c] += weight * d_colors[(ray * num_samples + s) * 3 + c];
            transmittance *= (1.0f - alpha);
            if (transmittance < 0.01f) break;
        }
        // Write predicted color
        for (int c = 0; c < 3; c++)
            d_pixel_colors[ray * 3 + c] = pixel_color[c];
        // Compute error and accumulate loss
        float pixel_loss = 0.0f;
        for (int c = 0; c < 3; c++) {
            float error = pixel_color[c] - d_true_colors[ray * num_samples * 3 + c];
            d_pixel_errors[ray * 3 + c] = error;
            pixel_loss += error * error;
        }
        atomicAdd(d_loss_accum, pixel_loss);
    }
}

// CUDA kernel for volume rendering gradient computation (for each ray)
__global__ void volume_rendering_gradient_kernel(
    const float* d_densities, const float* d_colors,
    const float* d_pixel_errors,
    float* d_mlp_error_output, int rays_per_batch, int num_samples)
{
    int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray < rays_per_batch) {
        float alphas[NUM_SAMPLES];
        float transmittance[NUM_SAMPLES];
        float weights[NUM_SAMPLES];
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
        for (int s = 0; s < num_samples; s++) {
            // Color gradients (sigmoid derivative)
            for (int c = 0; c < 3; c++) {
                float sigmoid_val = d_colors[(ray * num_samples + s) * 3 + c];
                float sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
                float grad = weights[s] * d_pixel_errors[ray * 3 + c] * sigmoid_deriv;
                d_mlp_error_output[(ray * num_samples + s) * 4 + 1 + c] = grad;
            }
            // Density gradient (ReLU derivative)
            float density = d_densities[ray * num_samples + s];
            float dalpha_ddensity = 0.01f * expf(-density * 0.01f);
            float density_gradient = dalpha_ddensity * transmittance[s] *
                (d_pixel_errors[ray * 3 + 0] * d_colors[(ray * num_samples + s) * 3 + 0] +
                 d_pixel_errors[ray * 3 + 1] * d_colors[(ray * num_samples + s) * 3 + 1] +
                 d_pixel_errors[ray * 3 + 2] * d_colors[(ray * num_samples + s) * 3 + 2]);
            for (int t = s + 1; t < num_samples; t++) {
                float dtrans_ddensity = -dalpha_ddensity;
                for (int k = s + 1; k <= t; k++)
                    dtrans_ddensity *= (1.0f - alphas[k - 1]);
                float dweight_t_ddensity = alphas[t] * dtrans_ddensity;
                density_gradient += dweight_t_ddensity *
                    (d_pixel_errors[ray * 3 + 0] * d_colors[(ray * num_samples + t) * 3 + 0] +
                     d_pixel_errors[ray * 3 + 1] * d_colors[(ray * num_samples + t) * 3 + 1] +
                     d_pixel_errors[ray * 3 + 2] * d_colors[(ray * num_samples + t) * 3 + 2]);
            }
            float relu_deriv = density > 0.0f ? 1.0f : 0.0f;
            d_mlp_error_output[(ray * num_samples + s) * 4 + 0] = density_gradient * relu_deriv;
        }
    }
}

// Full implementation of render_test_image
void render_test_image(MLP* mlp, Dataset* dataset, int batch_num, cublasHandle_t cublas_handle) {
    printf("  Rendering test image %dx%d...\n", RENDER_WIDTH, RENDER_HEIGHT);
    Camera* cam = dataset->cameras[0];
    MLP* temp_mlp = init_mlp(mlp->input_dim, mlp->hidden_dim, mlp->output_dim, NUM_SAMPLES, cublas_handle);
    cudaMemcpy(temp_mlp->d_W1, mlp->d_W1, mlp->hidden_dim * mlp->input_dim * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(temp_mlp->d_W2, mlp->d_W2, mlp->output_dim * mlp->hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(temp_mlp->d_W3, mlp->d_W3, mlp->output_dim * mlp->input_dim * sizeof(float), cudaMemcpyDeviceToDevice);

    float* ray_X = (float*)malloc(NUM_SAMPLES * RAW_INPUT_DIM * sizeof(float));
    float* ray_PE_X = (float*)malloc(NUM_SAMPLES * PE_INPUT_DIM * sizeof(float));
    unsigned char* image_data = (unsigned char*)malloc(RENDER_WIDTH * RENDER_HEIGHT * 3);

    // Buffers for CUDA pipeline
    float* d_ray_PE_X;
    float* d_layer2_output;
    float* d_densities;
    float* d_colors;
    cudaMalloc(&d_ray_PE_X, NUM_SAMPLES * PE_INPUT_DIM * sizeof(float));
    cudaMalloc(&d_layer2_output, NUM_SAMPLES * 4 * sizeof(float));
    cudaMalloc(&d_densities, NUM_SAMPLES * sizeof(float));
    cudaMalloc(&d_colors, NUM_SAMPLES * 3 * sizeof(float));

    for (int v = 0; v < RENDER_HEIGHT; v++) {
        for (int u = 0; u < RENDER_WIDTH; u++) {
            int scaled_u = (int)(u * (float)cam->width / RENDER_WIDTH);
            int scaled_v = (int)(v * (float)cam->height / RENDER_HEIGHT);
            float ray_o[3], ray_d[3];
            generate_ray(cam, scaled_u, scaled_v, ray_o, ray_d);
            for (int s = 0; s < NUM_SAMPLES; s++) {
                float t = NEAR_PLANE + (FAR_PLANE - NEAR_PLANE) * s / (NUM_SAMPLES - 1);
                ray_X[s * 6 + 0] = ray_o[0] + t * ray_d[0];
                ray_X[s * 6 + 1] = ray_o[1] + t * ray_d[1];
                ray_X[s * 6 + 2] = ray_o[2] + t * ray_d[2];
                ray_X[s * 6 + 3] = ray_d[0];
                ray_X[s * 6 + 4] = ray_d[1];
                ray_X[s * 6 + 5] = ray_d[2];
            }
            // Positional encoding
            batch_positional_encoding(ray_X, NUM_SAMPLES, 1, ray_PE_X);
            cudaMemcpy(d_ray_PE_X, ray_PE_X, NUM_SAMPLES * PE_INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);

            forward_pass_mlp(temp_mlp, d_ray_PE_X);

            int block_size = 64;
            int num_blocks = (NUM_SAMPLES + block_size - 1) / block_size;
            activation_kernel<<<num_blocks, block_size>>>(temp_mlp->d_layer2_preact, d_layer2_output, NUM_SAMPLES, temp_mlp->output_dim);
            extract_densities_colors_kernel<<<num_blocks, block_size>>>(d_layer2_output, d_densities, d_colors, NUM_SAMPLES, temp_mlp->output_dim);

            // Copy to host for rendering
            float densities[NUM_SAMPLES], colors[NUM_SAMPLES * 3];
            cudaMemcpy(densities, d_densities, NUM_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(colors, d_colors, NUM_SAMPLES * 3 * sizeof(float), cudaMemcpyDeviceToHost);

            float pixel_color[3] = {0,0,0};
            float transmittance = 1.0f;
            for (int s = 0; s < NUM_SAMPLES; s++) {
                float alpha = 1.0f - expf(-densities[s] * 0.01f);
                float weight = alpha * transmittance;
                for (int c = 0; c < 3; c++)
                    pixel_color[c] += weight * colors[s * 3 + c];
                transmittance *= (1.0f - alpha);
                if (transmittance < 0.01f) break;
            }
            int pixel_idx = (v * RENDER_WIDTH + u) * 3;
            for (int c = 0; c < 3; c++)
                image_data[pixel_idx + c] = (unsigned char)(fminf(1.0f, fmaxf(0.0f, pixel_color[c])) * 255);
        }
        if ((v + 1) % (RENDER_HEIGHT / 10) == 0)
            printf("    %d%% complete\n", (v + 1) * 100 / RENDER_HEIGHT);
    }
    char png_filename[128];
    snprintf(png_filename, sizeof(png_filename), "%06d_sample.png", batch_num);
    save_png(png_filename, image_data, RENDER_WIDTH, RENDER_HEIGHT);
    printf("  Test image saved as %s\n\n", png_filename);

    free(image_data);
    free(ray_X);
    free(ray_PE_X);
    free_mlp(temp_mlp);
    cudaFree(d_ray_PE_X);
    cudaFree(d_layer2_output);
    cudaFree(d_densities);
    cudaFree(d_colors);
}

int main() {
    srand(time(NULL));
    Dataset* dataset = load_dataset("./data/transforms.json", "./data", 100);

    const int input_dim = PE_INPUT_DIM; // Use encoded input dimension
    const int hidden_dim = 1024;
    const int output_dim = 4;
    const int batch_size = RAYS_PER_BATCH * NUM_SAMPLES;

    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size, cublas_handle);

    // Allocate host and device memory for batch
    float* batch_X = (float*)malloc(batch_size * RAW_INPUT_DIM * sizeof(float));
    float* batch_PE_X = (float*)malloc(batch_size * PE_INPUT_DIM * sizeof(float));
    float* batch_true_colors = (float*)malloc(batch_size * 3 * sizeof(float));
    float* d_batch_PE_X;
    float* d_batch_true_colors;
    cudaMalloc(&d_batch_PE_X, batch_size * PE_INPUT_DIM * sizeof(float));
    cudaMalloc(&d_batch_true_colors, batch_size * 3 * sizeof(float));

    // Device memory for activations, densities, colors, predicted pixel colors, pixel errors, loss
    float* d_layer2_output;
    float* d_densities;
    float* d_colors;
    float* d_pixel_colors;
    float* d_pixel_errors;
    float* d_loss_accum;
    cudaMalloc(&d_layer2_output, batch_size * output_dim * sizeof(float));
    cudaMalloc(&d_densities, batch_size * sizeof(float));
    cudaMalloc(&d_colors, batch_size * 3 * sizeof(float));
    cudaMalloc(&d_pixel_colors, RAYS_PER_BATCH * 3 * sizeof(float));
    cudaMalloc(&d_pixel_errors, RAYS_PER_BATCH * 3 * sizeof(float));
    cudaMalloc(&d_loss_accum, sizeof(float));

    float* d_mlp_error_output = mlp->d_error_output; // Use directly

    const int num_batches = 20000;
    float learning_rate = 0.001f;

    for (int batch = 0; batch < num_batches; batch++) {
        generate_random_batch(dataset, RAYS_PER_BATCH, batch_X, batch_true_colors);
        batch_positional_encoding(batch_X, NUM_SAMPLES, RAYS_PER_BATCH, batch_PE_X);
        cudaMemcpy(d_batch_PE_X, batch_PE_X, batch_size * PE_INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_batch_true_colors, batch_true_colors, batch_size * 3 * sizeof(float), cudaMemcpyHostToDevice);

        // Forward pass
        forward_pass_mlp(mlp, d_batch_PE_X);

        // Activation (on GPU)
        int block_size = 256;
        int num_blocks = (batch_size + block_size - 1) / block_size;
        activation_kernel<<<num_blocks, block_size>>>(mlp->d_layer2_preact, d_layer2_output, batch_size, output_dim);

        // Extract densities/colors
        extract_densities_colors_kernel<<<num_blocks, block_size>>>(d_layer2_output, d_densities, d_colors, batch_size, output_dim);

        // Zero gradients
        zero_gradients_mlp(mlp);

        // Volume rendering and loss computation (per ray)
        cudaMemset(d_loss_accum, 0, sizeof(float));
        int ray_block_size = 64;
        int ray_num_blocks = (RAYS_PER_BATCH + ray_block_size - 1) / ray_block_size;
        volume_rendering_and_loss_kernel<<<ray_num_blocks, ray_block_size>>>(
            d_densities, d_colors, d_batch_true_colors,
            d_pixel_colors, d_pixel_errors, d_loss_accum, RAYS_PER_BATCH, NUM_SAMPLES);

        // Volume rendering gradient computation (per ray)
        volume_rendering_gradient_kernel<<<ray_num_blocks, ray_block_size>>>(
            d_densities, d_colors, d_pixel_errors,
            d_mlp_error_output, RAYS_PER_BATCH, NUM_SAMPLES);

        // Backward pass
        backward_pass_mlp(mlp, d_batch_PE_X);
        update_weights_mlp(mlp, learning_rate);

        // Copy loss to host and print
        float total_loss = 0.0f;
        cudaMemcpy(&total_loss, d_loss_accum, sizeof(float), cudaMemcpyDeviceToHost);
        total_loss /= RAYS_PER_BATCH;

        if ((batch + 1) % 100 == 0) {
            printf("Batch [%d/%d], Loss: %.6f\n", batch + 1, num_batches, total_loss);
            if ((batch + 1) % 5000 == 0) render_test_image(mlp, dataset, batch + 1, cublas_handle);
        }
    }

    char model_filename[64];
    time_t now = time(NULL);
    strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    save_mlp(mlp, model_filename);

    printf("\nTraining completed! Model saved to: %s\n", model_filename);

    free_dataset(dataset);
    free(batch_X);
    free(batch_PE_X);
    free(batch_true_colors);
    free_mlp(mlp);
    cudaFree(d_batch_PE_X);
    cudaFree(d_batch_true_colors);
    cudaFree(d_layer2_output);
    cudaFree(d_densities);
    cudaFree(d_colors);
    cudaFree(d_pixel_colors);
    cudaFree(d_pixel_errors);
    cudaFree(d_loss_accum);
    cublasDestroy(cublas_handle);

    return 0;
}