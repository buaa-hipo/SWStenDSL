CC = mpicc
AR = ar

SRC = $(wildcard *.c)
OBJ = $(patsubst %.c, %.o, $(SRC))

all: libswstenmpi.a

.PHONY: clean

libswstenmpi.a: $(OBJ)
	$(AR) -rv $@ $^

%.o:%.c
	$(CC) -O2 -c $^ -o $@

clean:
	rm *.o *.a