CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto -lpng -ljson-c

ARCH ?= sm_86
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

mlp/gpu/mlp.o:
	$(MAKE) -C mlp/gpu mlp.o ARCH=$(ARCH)

run: train.out
	@time ./train.out

cont: train.out
	@time ./train.out $(shell ls -t *_nerf_model1.bin 2>/dev/null | head -1) $(shell ls -t *_nerf_model2.bin 2>/dev/null | head -1)

clean:
	rm -f *.out *.o *_sample.png
	$(MAKE) -C mlp clean