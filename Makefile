CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto -lpng -ljson-c

data.out: data.o
	$(CC) data.o $(LDFLAGS) -o $@

data.o: data.c
	$(CC) $(CFLAGS) -c data.c -o $@

data: data.out
	@time ./data.out

clean:
	rm -f *.out *.o *.csv
	$(MAKE) -C mlp clean