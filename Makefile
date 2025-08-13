CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto

ARCH ?= sm_87
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

# Source files
MLP_GPU_SOURCES = mlp/gpu/mlp.c mlp/data.c
NERF_SOURCES = nerf.c train_nerf.c

# Object files
MLP_GPU_OBJECTS = $(MLP_GPU_SOURCES:.c=.o)
NERF_OBJECTS = $(NERF_SOURCES:.c=.o)

# Targets
all: train_nerf.out

train_nerf.out: $(MLP_GPU_OBJECTS) $(NERF_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(CUDALIBS) $(LDFLAGS)

# Pattern rule for GPU MLP objects
mlp/gpu/%.o: mlp/gpu/%.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c $< -o $@

# Pattern rule for regular objects  
mlp/%.o: mlp/%.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c $< -o $@

# Pattern rule for NeRF objects
%.o: %.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c $< -o $@

clean:
	rm -f $(MLP_GPU_OBJECTS) $(NERF_OBJECTS) train_nerf.out *.bin

run: train_nerf.out
	time ./train_nerf.out

.PHONY: all clean run