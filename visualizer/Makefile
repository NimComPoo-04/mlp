CC = gcc
CFLAGS = -Wall -Wextra -ggdb -O0 -std=c11

# change this variable to the location where raylib is located
RAYLIB = ~/opt/raylib-5.0_linux_amd64

libnnvis.a: nnvis.o ../mlp.o
	make -C ..
	ar rs $@ $^

nnvis.o: nnvis.c
	$(CC) $(CFLAGS) -static -c -o $@ $^ -I $(RAYLIB)/include -I ..

test: test.c libnnvis.a
	$(CC) $(CFLAGS) -o $@ $< -I $(RAYLIB)/include -I .. -L $(RAYLIB)/lib -L . -lnnvis -lm -lraylib
	export LD_LIBRARY_PATH=$(RAYLIB)/lib ; ./test

clean:
	rm -rf *.o *.a test
