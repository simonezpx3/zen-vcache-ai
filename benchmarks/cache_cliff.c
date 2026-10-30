#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <sys/mman.h>

#define CACHE_LINE_SIZE 64
#define ACCESS_COUNT 20000000ULL

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static uint32_t lcg_rand(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static double measure_latency(size_t size_bytes, int use_hugepages) {
    size_t num_lines = size_bytes / CACHE_LINE_SIZE;
    if (num_lines < 2) return 0.0;

    size_t *indices = (size_t *)malloc(num_lines * sizeof(size_t));
    if (!indices) {
        perror("malloc indices");
        return -1.0;
    }
    for (size_t i = 0; i < num_lines; i++) {
        indices[i] = i;
    }

    uint32_t rng = (uint32_t)(0x732641 ^ size_bytes);
    for (size_t i = num_lines - 1; i > 0; i--) {
        size_t j = lcg_rand(&rng) % (i + 1);
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    void *mem = mmap(NULL, size_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap");
        free(indices);
        return -1.0;
    }

    if (use_hugepages) {
        madvise(mem, size_bytes, MADV_HUGEPAGE);
    }

    uintptr_t base = (uintptr_t)mem;
    for (size_t i = 0; i < num_lines - 1; i++) {
        uintptr_t curr_addr = base + indices[i] * CACHE_LINE_SIZE;
        uintptr_t next_addr = base + indices[i + 1] * CACHE_LINE_SIZE;
        *(uintptr_t *)curr_addr = next_addr;
    }
    uintptr_t last_addr = base + indices[num_lines - 1] * CACHE_LINE_SIZE;
    *(uintptr_t *)last_addr = base + indices[0] * CACHE_LINE_SIZE;

    uintptr_t first_addr = base + indices[0] * CACHE_LINE_SIZE;
    free(indices);

    volatile uintptr_t *p = (volatile uintptr_t *)first_addr;
    for (size_t i = 0; i < 500000; i++) {
        p = (volatile uintptr_t *)(*p);
    }

    uint64_t start_ns = get_time_ns();
    for (uint64_t i = 0; i < ACCESS_COUNT; i++) {
        p = (volatile uintptr_t *)(*p);
    }
    uint64_t end_ns = get_time_ns();

    munmap(mem, size_bytes);

    if (p == (volatile uintptr_t *)0x1) {
        printf("dummy: %p\n", (void *)p);
    }

    double elapsed_ns = (double)(end_ns - start_ns);
    return elapsed_ns / (double)ACCESS_COUNT;
}

int main(int argc, char **argv) {
    size_t test_sizes_mb[] = { 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 256 };
    size_t count = sizeof(test_sizes_mb) / sizeof(test_sizes_mb[0]);

    printf("========================================================================================\n");
    printf(" AMD Ryzen 7 7800X3D Hardware Cache Cliff Benchmark (Pointer Chasing)\n");
    printf(" 96MB L3 SRAM vs System DDR5 Latency Boundary (4KB Pages vs 2MB HugePages)\n");
    printf("========================================================================================\n");
    printf("%-12s | %-16s | %-16s | %-28s\n", "Buffer Size", "4KB Latency", "HugePage Latency", "Residency Zone");
    printf("-------------+------------------+------------------+-----------------------------\n");

    for (size_t i = 0; i < count; i++) {
        size_t mb = test_sizes_mb[i];
        size_t bytes = mb * 1024ULL * 1024ULL;
        double lat_4k = measure_latency(bytes, 0);
        double lat_huge = measure_latency(bytes, 1);

        const char *zone;
        if (mb <= 32) {
            zone = "L3 CCD Base (Standard Zen 4)";
        } else if (mb <= 96) {
            zone = "3D V-Cache (Stacked SRAM)";
        } else {
            zone = "DDR5 Memory Bus (Spillover)";
        }

        printf("%4zu MB      | %12.2f ns   | %12.2f ns   | %s\n", mb, lat_4k, lat_huge, zone);
        fflush(stdout);
    }
    printf("========================================================================================\n");
    return 0;
}
