CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto -lpng -ljson-c -lopenblas

train.out: train.o mlp.o data.o
	$(CC) train.o mlp.o data.o $(LDFLAGS) -o $@

train.o: train.c mlp/mlp.h data.h
	$(CC) $(CFLAGS) -c train.c -o $@

mlp.o: mlp/mlp.c mlp/mlp.h
	$(CC) $(CFLAGS) -c mlp/mlp.c -o $@

data.o: data.c data.h
	$(CC) $(CFLAGS) -c data.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *_sample.png
	$(MAKE) -C mlp clean