#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>

#define DEFAULT_CACHE_LINE_SIZE 64
#define HUGEPAGE_SIZE (2ULL * 1024ULL * 1024ULL)
#define ACCESS_COUNT 20000000ULL
#define REPETITIONS 5

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline uint32_t lcg_rand(uint32_t *state) {
    *state = (*state) * 1664525u + 1013904223u;
    return *state;
}

static void pin_to_cpu(int cpu_id) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("Warning: sched_setaffinity failed");
    }
}

static void *alloc_aligned_buffer(size_t size_bytes, bool use_hugepages) {
    size_t alloc_size = size_bytes + HUGEPAGE_SIZE;
    void *raw = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        return MAP_FAILED;
    }

    uintptr_t raw_addr = (uintptr_t)raw;
    uintptr_t aligned_addr = (raw_addr + HUGEPAGE_SIZE - 1) & ~(HUGEPAGE_SIZE - 1);

    size_t prefix = aligned_addr - raw_addr;
    if (prefix > 0) {
        munmap(raw, prefix);
    }
    size_t suffix = alloc_size - prefix - size_bytes;
    if (suffix > 0) {
        munmap((void *)(aligned_addr + size_bytes), suffix);
    }

    void *mem = (void *)aligned_addr;
    if (use_hugepages) {
        if (madvise(mem, size_bytes, MADV_HUGEPAGE) != 0) {
            perror("Warning: madvise(MADV_HUGEPAGE) failed");
        }
    } else {
        madvise(mem, size_bytes, MADV_NOHUGEPAGE);
    }

    return mem;
}

static double measure_latency(size_t size_bytes, bool use_hugepages, size_t line_size) {
    size_t num_lines = size_bytes / line_size;
    if (num_lines < 2) return NAN;

    size_t *indices = malloc(num_lines * sizeof(size_t));
    if (!indices) {
        perror("malloc indices");
        return NAN;
    }
    for (size_t i = 0; i < num_lines; i++) {
        indices[i] = i;
    }

    uint32_t rng = (uint32_t)(0x732641ULL ^ size_bytes);
    for (size_t i = num_lines - 1; i > 0; i--) {
        size_t j = lcg_rand(&rng) % i;
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    void *mem = alloc_aligned_buffer(size_bytes, use_hugepages);
    if (mem == MAP_FAILED) {
        perror("mmap aligned");
        free(indices);
        return NAN;
    }

    uintptr_t base = (uintptr_t)mem;
    for (size_t i = 0; i < num_lines - 1; i++) {
        uintptr_t curr_addr = base + indices[i] * line_size;
        uintptr_t next_addr = base + indices[i + 1] * line_size;
        *(uintptr_t *)curr_addr = next_addr;
    }
    uintptr_t last_addr = base + indices[num_lines - 1] * line_size;
    uintptr_t first_addr = base + indices[0] * line_size;
    *(uintptr_t *)last_addr = first_addr;

    free(indices);

    size_t warmup_iters = num_lines * 2;
    if (warmup_iters < 1000000ULL) warmup_iters = 1000000ULL;

    volatile uintptr_t *p = (volatile uintptr_t *)first_addr;
    for (size_t i = 0; i < warmup_iters; i++) {
        p = (volatile uintptr_t *)(*p);
    }

    asm volatile("" ::: "memory");
    uint64_t start_ns = get_time_ns();
    asm volatile("" ::: "memory");

    for (uint64_t i = 0; i < ACCESS_COUNT; i++) {
        p = (volatile uintptr_t *)(*p);
    }

    asm volatile("" ::: "memory");
    uint64_t end_ns = get_time_ns();
    asm volatile("" ::: "memory");

    munmap(mem, size_bytes);

    if (p == (volatile uintptr_t *)0x1) {
        printf("dummy: %p\n", (void *)p);
    }

    return (double)(end_ns - start_ns) / (double)ACCESS_COUNT;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(void) {
    pin_to_cpu(0);

    long sys_line_sz = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    size_t line_size = (sys_line_sz > 0) ? (size_t)sys_line_sz : DEFAULT_CACHE_LINE_SIZE;

    const size_t test_sizes_mb[] = { 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 256 };
    size_t count = sizeof(test_sizes_mb) / sizeof(test_sizes_mb[0]);

    printf("====================================================================================================\n");
    printf(" AMD Ryzen 7 7800X3D Cache & TLB Cliff Benchmark (Pointer Chasing, CPU Pinned, 2MB Aligned)\n");
    printf(" Unified 96MB L3 (3D V-Cache) vs System DDR5 | 4KB Pages (TLB-bound) vs 2MB THP (Cache-bound)\n");
    printf("====================================================================================================\n");
    printf("%-10s | %-22s | %-22s | %-30s\n", "Size", "4KB Latency (med±std)", "2MB THP (med±std)", "Primary Bottleneck / Zone");
    printf("-----------+------------------------+------------------------+--------------------------------------\n");

    for (size_t i = 0; i < count; i++) {
        size_t mb = test_sizes_mb[i];
        size_t bytes = mb * 1024ULL * 1024ULL;

        double samples_4k[REPETITIONS];
        double samples_thp[REPETITIONS];
        double sum_4k = 0.0, sq_4k = 0.0;
        double sum_thp = 0.0, sq_thp = 0.0;

        for (size_t r = 0; r < REPETITIONS; r++) {
            samples_4k[r] = measure_latency(bytes, false, line_size);
            samples_thp[r] = measure_latency(bytes, true, line_size);

            if (isnan(samples_4k[r]) || isnan(samples_thp[r])) {
                fprintf(stderr, "Fatal error allocating memory for %zu MB. Aborting.\n", mb);
                return EXIT_FAILURE;
            }

            sum_4k += samples_4k[r];
            sq_4k += samples_4k[r] * samples_4k[r];
            sum_thp += samples_thp[r];
            sq_thp += samples_thp[r] * samples_thp[r];
        }

        qsort(samples_4k, REPETITIONS, sizeof(double), cmp_double);
        qsort(samples_thp, REPETITIONS, sizeof(double), cmp_double);

        double med_4k = samples_4k[REPETITIONS / 2];
        double med_thp = samples_thp[REPETITIONS / 2];

        double avg_4k = sum_4k / REPETITIONS;
        double std_4k = sqrt(fmax(0.0, (sq_4k / REPETITIONS) - (avg_4k * avg_4k)));

        double avg_thp = sum_thp / REPETITIONS;
        double std_thp = sqrt(fmax(0.0, (sq_thp / REPETITIONS) - (avg_thp * avg_thp)));

        const char *zone;
        if (mb <= 12) {
            zone = "L3 SRAM (Fits in L2 dTLB)";
        } else if (mb <= 96) {
            zone = "Unified 96MB L3 (4KB = TLB Misses)";
        } else {
            zone = "DDR5 Memory Bus (L3 Spillover)";
        }

        printf("%4zu MB    | %9.2f ± %5.2f ns   | %9.2f ± %5.2f ns   | %s\n",
               mb, med_4k, std_4k, med_thp, std_thp, zone);
        fflush(stdout);
    }
    printf("====================================================================================================\n");
    return EXIT_SUCCESS;
}
