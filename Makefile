CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto -lpng -ljson-c

ARCH ?= sm_87
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

train.out: train.o nerf.o data.o mlp/gpu/mlp.o
	$(CC) train.o nerf.o data.o mlp/gpu/mlp.o $(CUDALIBS) $(LDFLAGS) -o $@

train.o: train.c nerf.h data.h mlp/gpu/mlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

nerf.o: nerf.c nerf.h data.h mlp/gpu/mlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c nerf.c -o $@

data.o: data.c data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c data.c -o data.o

mlp/gpu/mlp.o: mlp/gpu/mlp.c mlp/gpu/mlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c mlp/gpu/mlp.c -o mlp/gpu/mlp.o

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *_sample.png
	$(MAKE) -C mlp clean