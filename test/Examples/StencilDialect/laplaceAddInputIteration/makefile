CC = sw5cc
CFLAGS = -O2

SRC = $(wildcard *.c)
OBJ = $(patsubst %.c, %.o, $(SRC))

.PHONY: clean run

exe: $(OBJ)
	$(CC) -hybrid $(CFLAGS) $^ -o $@

%_slave.o: %_slave.c
	$(CC) -slave $(CFLAGS) -c $^ -o $@

%.o: %.c
	$(CC) -host $(CFLAGS) -c $^ -o $@

run:
	bsub -I -b -q q_sw_expr -n 1 -cgsp 64 -share_size 2048 ./exe

clean:
	rm -f *.o exe mpe_result spe_result