CC = gcc
CFLAGS = -Wall -Wextra -ggdb -std=c11 -O0

all: binadd sine

sine:
	$(CC) $(CFLAGS) -I.. -o $@ sine.c ../mlp.o -lm
	./sine 2> img.ppm

binadd:
	$(CC) $(CFLAGS) -I.. -o $@ binadd.c ../mlp.o -lm
	./binadd

.PHONY: all sine binadd
