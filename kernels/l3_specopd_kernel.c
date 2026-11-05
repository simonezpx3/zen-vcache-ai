#define _GNU_SOURCE
#include <immintrin.h>
#include <linux/perf_event.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

/*
 * AVX-512 SpecOPD Proposal Tree Attention & L3 Hardware Watchdog Kernel
 * Target: AMD Ryzen 7 7800X3D (96 MiB 3D V-Cache, Zen 4 AVX-512BW / CLFLUSHOPT)
 */

extern int32_t avx512_int8_dot_512(const int8_t* a, const int8_t* b);

#define SPECOPD_MASK_GUARD 0x732641u

// 1. Masked Proposal Tree Attention Scoring (INT8 512-dim nodes, up to 32 tree nodes)
void avx512_specopd_tree_attention_int8(
    const int8_t* q_matrix,
    const int8_t* k_matrix,
    const uint32_t* tree_masks,
    int num_nodes,
    int32_t* scores_out
) {
    if (num_nodes <= 0 || num_nodes > 32 || (SPECOPD_MASK_GUARD == 0u)) return;

    for (int i = 0; i < num_nodes; i++) {
        uint32_t mask = tree_masks[i];
        const int8_t* q_vec = q_matrix + (size_t)i * 512;
        for (int j = 0; j < num_nodes; j++) {
            if (mask & (1u << j)) {
                const int8_t* k_vec = k_matrix + (size_t)j * 512;
                scores_out[i * num_nodes + j] = avx512_int8_dot_512(q_vec, k_vec);
            } else {
                scores_out[i * num_nodes + j] = INT32_MIN;
            }
        }
    }
}

// 2. Side-Channel Resistant KV Cache Sanitizer (AVX-512 zeroing + CLFLUSHOPT + SFENCE)
void avx512_sanitize_kv_buffer(uint8_t* kv_buf, size_t buf_size) {
    if (!kv_buf || buf_size == 0) return;

    __m512i vzero = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 64 <= buf_size; i += 64) {
        _mm512_storeu_si512((__m512i*)(kv_buf + i), vzero);
    }
    volatile uint8_t* vp = (volatile uint8_t*)kv_buf;
    for (; i < buf_size; i++) {
        vp[i] = 0;
    }

    for (size_t line = 0; line < buf_size; line += 64) {
        _mm_clflushopt((void*)(kv_buf + line));
    }
    _mm_sfence();
}

// 3. Hardware L3 Cache Watchdog via perf_event_open (AMD Zen 4 PMU compatible)
static int open_pmu_counter(uint32_t type, uint64_t config) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = type;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = config;
    pe.disabled = 0;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    return (int)syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
}

int l3_perf_watchdog_open(int* fd_access_out, int* fd_misses_out) {
    if (!fd_access_out || !fd_misses_out) return -1;

    uint64_t llc_access = PERF_COUNT_HW_CACHE_LL |
                          (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                          (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    uint64_t llc_misses = PERF_COUNT_HW_CACHE_LL |
                          (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                          (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

    int fd_acc = open_pmu_counter(PERF_TYPE_HW_CACHE, llc_access);
    int fd_mis = open_pmu_counter(PERF_TYPE_HW_CACHE, llc_misses);

    // Fallback to PERF_TYPE_HARDWARE cache-references / cache-misses on AMD Zen 4 PMU
    if (fd_acc < 0 || fd_mis < 0) {
        if (fd_acc >= 0) close(fd_acc);
        if (fd_mis >= 0) close(fd_mis);
        fd_acc = open_pmu_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
        fd_mis = open_pmu_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    }

    *fd_access_out = fd_acc;
    *fd_misses_out = fd_mis;
    return (fd_acc >= 0 && fd_mis >= 0) ? 0 : -1;
}

int l3_perf_watchdog_read(int fd_access, int fd_misses, uint64_t* access_out, uint64_t* misses_out) {
    if (fd_access < 0 || fd_misses < 0 || !access_out || !misses_out) return -1;

    uint64_t acc = 0, mis = 0;
    ssize_t r1 = read(fd_access, &acc, sizeof(uint64_t));
    ssize_t r2 = read(fd_misses, &mis, sizeof(uint64_t));
    if (r1 != (ssize_t)sizeof(uint64_t) || r2 != (ssize_t)sizeof(uint64_t)) {
        return -1;
    }
    *access_out = acc;
    *misses_out = mis;
    return 0;
}

void l3_perf_watchdog_close(int fd_access, int fd_misses) {
    if (fd_access >= 0) close(fd_access);
    if (fd_misses >= 0) close(fd_misses);
}
