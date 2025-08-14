#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "mlp/mlp.h"
#include "data.h"

#define RAYS_PER_BATCH 8
#define RENDER_WIDTH 128
#define RENDER_HEIGHT 128

// Activation functions
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static inline float sigmoid_derivative(float sigmoid_val) {
    return sigmoid_val * (1.0f - sigmoid_val);
}
static inline float relu(float x) {
    return fmaxf(0.0f, x);
}
static inline float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// Volume rendering: integrate densities and colors along ray
void render_ray(const float* densities, const float* colors, float* pixel_color) {
    pixel_color[0] = pixel_color[1] = pixel_color[2] = 0.0f;
    float transmittance = 1.0f;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        float alpha = 1.0f - expf(-densities[i] * 0.01f);
        float weight = alpha * transmittance;
        pixel_color[0] += weight * colors[i * 3 + 0];
        pixel_color[1] += weight * colors[i * 3 + 1];
        pixel_color[2] += weight * colors[i * 3 + 2];
        transmittance *= (1.0f - alpha);
        if (transmittance < 0.01f) break;
    }
}

// Compute volume rendering gradients
void compute_volume_rendering_gradients(const float* densities, const float* colors, const float* pixel_error, float* mlp_error_output) {
    float alphas[NUM_SAMPLES], transmittance[NUM_SAMPLES], weights[NUM_SAMPLES];
    for (int s = 0; s < NUM_SAMPLES; s++) alphas[s] = 1.0f - expf(-densities[s] * 0.01f);

    transmittance[0] = 1.0f;
    weights[0] = alphas[0] * transmittance[0];
    for (int s = 1; s < NUM_SAMPLES; s++) {
        transmittance[s] = transmittance[s-1] * (1.0f - alphas[s-1]);
        weights[s] = alphas[s] * transmittance[s];
    }

    for (int s = 0; s < NUM_SAMPLES; s++) {
        // Color gradients with sigmoid derivative
        for (int c = 0; c < 3; c++) {
            float sigmoid_val = colors[s * 3 + c];
            float sigmoid_deriv = sigmoid_derivative(sigmoid_val);
            mlp_error_output[s * 4 + 1 + c] = weights[s] * pixel_error[c] * sigmoid_deriv;
        }
        // Density gradient with ReLU derivative
        float dalpha_ddensity = 0.01f * expf(-densities[s] * 0.01f);
        float density_gradient = dalpha_ddensity * transmittance[s] * (pixel_error[0] * colors[s * 3 + 0] + pixel_error[1] * colors[s * 3 + 1] + pixel_error[2] * colors[s * 3 + 2]);
        for (int t = s + 1; t < NUM_SAMPLES; t++) {
            float dtrans_ddensity = -dalpha_ddensity;
            for (int k = s + 1; k <= t; k++) dtrans_ddensity *= (1.0f - alphas[k-1]);
            float dweight_t_ddensity = alphas[t] * dtrans_ddensity;
            density_gradient += dweight_t_ddensity * (pixel_error[0] * colors[t * 3 + 0] + pixel_error[1] * colors[t * 3 + 1] + pixel_error[2] * colors[t * 3 + 2]);
        }
        float relu_deriv = relu_derivative(densities[s]);
        mlp_error_output[s * 4 + 0] = density_gradient * relu_deriv;
    }
}

// Apply activations to layer2_output
void apply_activations_mlp(const MLP* mlp, float* layer2_output) {
    for (int i = 0; i < mlp->batch_size; i++) {
        int base = i * mlp->output_dim;
        layer2_output[base + 0] = relu(mlp->layer2_preact[base + 0]); // Density
        layer2_output[base + 1] = sigmoid(mlp->layer2_preact[base + 1]);
        layer2_output[base + 2] = sigmoid(mlp->layer2_preact[base + 2]);
        layer2_output[base + 3] = sigmoid(mlp->layer2_preact[base + 3]);
    }
}

// Extract densities and colors for a ray from layer2_output
void extract_densities_colors(const float* layer2_output, int ray_start_idx, float* densities, float* colors) {
    for (int s = 0; s < NUM_SAMPLES; s++) {
        int base = ray_start_idx * 4 + s * 4;
        densities[s] = layer2_output[base + 0];
        colors[s * 3 + 0] = layer2_output[base + 1];
        colors[s * 3 + 1] = layer2_output[base + 2];
        colors[s * 3 + 2] = layer2_output[base + 3];
    }
}

// Process batch and compute loss
float process_batch(MLP* mlp, const float* layer2_output, const float* batch_true_colors) {
    float total_loss = 0.0f;
    for (int ray = 0; ray < RAYS_PER_BATCH; ray++) {
        int ray_start_idx = ray * NUM_SAMPLES;
        float* ray_error_output = &mlp->error_output[ray_start_idx * 4];
        float densities[NUM_SAMPLES], colors[NUM_SAMPLES * 3];
        extract_densities_colors(layer2_output, ray_start_idx, densities, colors);

        float predicted_pixel_color[3];
        render_ray(densities, colors, predicted_pixel_color);

        const float* true_pixel_color = &batch_true_colors[ray_start_idx * 3];
        float pixel_error[3];
        float pixel_loss = 0.0f;
        for (int c = 0; c < 3; c++) {
            pixel_error[c] = predicted_pixel_color[c] - true_pixel_color[c];
            pixel_loss += pixel_error[c] * pixel_error[c];
        }
        total_loss += pixel_loss;
        compute_volume_rendering_gradients(densities, colors, pixel_error, ray_error_output);
    }
    return total_loss / RAYS_PER_BATCH;
}

// Render a single pixel
void render_single_pixel(MLP* temp_mlp, float* ray_X, Camera* cam, int u, int v, float* pixel_color) {
    float ray_o[3], ray_d[3];
    generate_ray(cam, u, v, ray_o, ray_d);
    for (int s = 0; s < NUM_SAMPLES; s++) {
        float t = NEAR_PLANE + (FAR_PLANE - NEAR_PLANE) * s / (NUM_SAMPLES - 1);
        ray_X[s * 6 + 0] = ray_o[0] + t * ray_d[0];
        ray_X[s * 6 + 1] = ray_o[1] + t * ray_d[1];
        ray_X[s * 6 + 2] = ray_o[2] + t * ray_d[2];
        ray_X[s * 6 + 3] = ray_d[0];
        ray_X[s * 6 + 4] = ray_d[1];
        ray_X[s * 6 + 5] = ray_d[2];
    }
    forward_pass_mlp(temp_mlp, ray_X);
    float* layer2_output = (float*)malloc(NUM_SAMPLES * 4 * sizeof(float));
    apply_activations_mlp(temp_mlp, layer2_output);
    float densities[NUM_SAMPLES], colors[NUM_SAMPLES * 3];
    extract_densities_colors(layer2_output, 0, densities, colors);
    render_ray(densities, colors, pixel_color);
    free(layer2_output);
}

// Render a test image
void render_test_image(MLP* mlp, Dataset* dataset, int batch_num) {
    printf("  Rendering test image %dx%d...\n", RENDER_WIDTH, RENDER_HEIGHT);
    Camera* cam = dataset->cameras[0];
    MLP* temp_mlp = init_mlp(mlp->input_dim, mlp->hidden_dim, mlp->output_dim, NUM_SAMPLES);
    memcpy(temp_mlp->W1, mlp->W1, mlp->hidden_dim * mlp->input_dim * sizeof(float));
    memcpy(temp_mlp->W2, mlp->W2, mlp->output_dim * mlp->hidden_dim * sizeof(float));
    memcpy(temp_mlp->W3, mlp->W3, mlp->output_dim * mlp->input_dim * sizeof(float));
    float* ray_X = (float*)malloc(NUM_SAMPLES * 6 * sizeof(float));
    unsigned char* image_data = (unsigned char*)malloc(RENDER_WIDTH * RENDER_HEIGHT * 3);

    for (int v = 0; v < RENDER_HEIGHT; v++) {
        for (int u = 0; u < RENDER_WIDTH; u++) {
            int scaled_u = (int)(u * (float)cam->width / RENDER_WIDTH);
            int scaled_v = (int)(v * (float)cam->height / RENDER_HEIGHT);
            float pixel_color[3];
            render_single_pixel(temp_mlp, ray_X, cam, scaled_u, scaled_v, pixel_color);
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
    free_mlp(temp_mlp);
}

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);

    Dataset* dataset = load_dataset("./data/transforms.json", "./data", 100);

    const int input_dim = 6;
    const int hidden_dim = 1024;
    const int output_dim = 4;
    const int batch_size = RAYS_PER_BATCH * NUM_SAMPLES;

    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size);
    float* batch_X = (float*)malloc(batch_size * input_dim * sizeof(float));
    float* batch_true_colors = (float*)malloc(batch_size * 3 * sizeof(float));
    float* layer2_output = (float*)malloc(batch_size * output_dim * sizeof(float));

    const int num_batches = 20000;
    float learning_rate = 0.001f;

    for (int batch = 0; batch < num_batches; batch++) {
        generate_random_batch(dataset, RAYS_PER_BATCH, batch_X, batch_true_colors);
        forward_pass_mlp(mlp, batch_X);
        apply_activations_mlp(mlp, layer2_output);
        zero_gradients_mlp(mlp);
        float loss = process_batch(mlp, layer2_output, batch_true_colors);
        backward_pass_mlp(mlp, batch_X);
        update_weights_mlp(mlp, learning_rate);

        if ((batch + 1) % 100 == 0) {
            printf("Batch [%d/%d], Loss: %.6f\n", batch + 1, num_batches, loss);
            if ((batch + 1) % 5000 == 0) render_test_image(mlp, dataset, batch + 1);
        }
    }
    char model_filename[64];
    time_t now = time(NULL);
    strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    save_mlp(mlp, model_filename);

    printf("\nTraining completed! Model saved to: %s\n", model_filename);

    free_dataset(dataset);
    free(batch_X);
    free(batch_true_colors);
    free(layer2_output);
    free_mlp(mlp);

    return 0;
}