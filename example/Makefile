CC = gcc
CFLAGS = -Wall -Wextra -ggdb -std=c11 -O0

all: binadd sine

sine: sine.c
	$(CC) $(CFLAGS) -I.. -o $@ $^ ../mlp.o -lm
	./sine 2> img.ppm

binadd: binadd.c
	$(CC) $(CFLAGS) -I.. -o $@ $^ ../mlp.o -lm
	./binadd

.PHONY: all
