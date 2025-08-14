CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto -lpng -ljson-c

ARCH ?= sm_87
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

train.out: train.o mlp/gpu/mlp.o mlp/data.o data.o
	$(CC) train.o mlp/gpu/mlp.o mlp/data.o data.o $(CUDALIBS) $(LDFLAGS) -o $@

train.o: train.c mlp/gpu/mlp.h data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

mlp/gpu/mlp.o: mlp/gpu/mlp.c mlp/gpu/mlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c mlp/gpu/mlp.c -o mlp/gpu/mlp.o

data.o: data.c data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c data.c -o data.o

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *_sample.png
	rm -f mlp/*.o mlp/gpu/*.o mlp/*.out mlp/*.bin mlp/*.csv mlp/gpu/*.out mlp/gpu/*.bin mlp/gpu/*.csv