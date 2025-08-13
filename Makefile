CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto

ARCH ?= sm_87
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

train.out: mlp.o nerf.o train.o
	$(CC) mlp.o nerf.o train.o $(CUDALIBS) $(LDFLAGS) -o $@

mlp.o: mlp/gpu/mlp.c mlp/gpu/mlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c mlp/gpu/mlp.c -o $@

nerf.o: nerf.c nerf.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c nerf.c -o $@

train.o: train.c nerf.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin