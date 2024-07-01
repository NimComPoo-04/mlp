CC = gcc
CFLAGS = -Wall -Wextra -ggdb -std=c11 -O0

mlp.o: mlp.c
	$(CC) $(CFLAGS) -c -o $@ $^

example:
	make -C example

clean:
	rm mlp.o

.PHONY: example clean
