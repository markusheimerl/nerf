CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto -lpng -ljson-c -lopenblas

train.out: train.o mlp.o data.o
	$(CC) train.o mlp.o $(LDFLAGS) -o $@

train.o: train.c mlp/mlp.h
	$(CC) $(CFLAGS) -c train.c -o $@

mlp.o: mlp/mlp.c mlp/mlp.h
	$(CC) $(CFLAGS) -c mlp/mlp.c -o $@

data.o: data.c
	$(CC) $(CFLAGS) -c data.c -o $@

data.out: data.o
	$(CC) data.o -lm -flto -lpng -ljson-c -o $@

data: data.out
	@time ./data.out

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o
	$(MAKE) -C mlp clean