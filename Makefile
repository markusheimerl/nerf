CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm

# Source files
MLP_SOURCES = mlp/mlp.c mlp/data.c
NERF_SOURCES = nerf.c train_nerf.c

# Object files
MLP_OBJECTS = $(MLP_SOURCES:.c=.o)
NERF_OBJECTS = $(NERF_SOURCES:.c=.o)

# Targets
all: train_nerf.out

train_nerf.out: $(MLP_OBJECTS) $(NERF_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(MLP_OBJECTS) $(NERF_OBJECTS) train_nerf.out *.bin

run: train_nerf.out
	time ./train_nerf.out

.PHONY: all clean run