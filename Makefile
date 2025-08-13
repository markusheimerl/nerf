CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto -lpng -ljson-c

ARCH ?= sm_87
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

train.out: mlp.o nerf.o train.o image_utils.o
	$(CC) mlp.o nerf.o train.o image_utils.o $(CUDALIBS) $(LDFLAGS) -o $@

mlp.o: mlp/gpu/mlp.c mlp/gpu/mlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c mlp/gpu/mlp.c -o $@

nerf.o: nerf.c nerf.h image_utils.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c nerf.c -o $@

train.o: train.c nerf.h image_utils.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

image_utils.o: image_utils.c image_utils.h
	$(CC) $(CFLAGS) -c image_utils.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin

.PHONY: run clean