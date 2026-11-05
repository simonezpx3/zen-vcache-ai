CC ?= gcc
CFLAGS ?= -O3 -march=znver4 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vpopcntdq -mavx512vnni -mavx512vbmi -fPIC -shared

all: libl3_vector.so

libl3_vector.so: l3_vector_kernel.c l3_specopd_kernel.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f libl3_vector.so

.PHONY: all clean
