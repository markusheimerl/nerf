CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto -lpng -ljson-c

ARCH ?= sm_87
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

train.out: nerf.o train.o
	$(CC) nerf.o train.o $(CUDALIBS) $(LDFLAGS) -o $@

nerf.o: nerf.c nerf.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c nerf.c -o $@

train.o: train.c nerf.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin

.PHONY: run clean