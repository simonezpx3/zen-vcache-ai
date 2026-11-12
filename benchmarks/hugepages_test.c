#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <sys/mman.h>

#define BUFFER_SIZE (64ULL * 1024ULL * 1024ULL) // 64 MB
#define ITERATIONS 100

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static double test_buffer(int use_hugepages) {
    void *mem = mmap(NULL, BUFFER_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap");
        return -1.0;
    }

    if (use_hugepages) {
        if (madvise(mem, BUFFER_SIZE, MADV_HUGEPAGE) != 0) {
            perror("madvise MADV_HUGEPAGE");
        }
    }

    // Touch all pages to fault them in
    memset(mem, 0x5A, BUFFER_SIZE);

    volatile uint64_t *ptr = (volatile uint64_t *)mem;
    size_t step = 4096 / sizeof(uint64_t); // 1 access per 4KB page
    size_t count = BUFFER_SIZE / sizeof(uint64_t);

    uint64_t start_ns = get_time_ns();
    uint64_t sum = 0;
    for (int it = 0; it < ITERATIONS; it++) {
        for (size_t i = 0; i < count; i += step) {
            sum += ptr[i];
        }
    }
    uint64_t end_ns = get_time_ns();

    munmap(mem, BUFFER_SIZE);

    if (sum == 0x12345678) {
        printf("dummy: %lu\n", sum);
    }

    double elapsed_ms = (double)(end_ns - start_ns) / 1000000.0;
    return elapsed_ms;
}

int main(void) {
    printf("========================================================================\n");
    printf(" 64MB L3 Working Buffer: 4KB Pages vs 2MB HugePages Latency\n");
    printf("========================================================================\n");

    // Warmup
    test_buffer(0);

    double ms_4k = test_buffer(0);
    double ms_huge = test_buffer(1);
    double delta = ((ms_4k - ms_huge) / ms_4k) * 100.0;

    printf("4KB Standard Pages : %8.2f ms\n", ms_4k);
    printf("2MB HugePages (THP): %8.2f ms\n", ms_huge);
    printf("Improvement        : %+7.1f %%\n", delta);
    printf("========================================================================\n");

    return 0;
}
