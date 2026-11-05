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

// 3. Hardware L3 Cache Watchdog via perf_event_open (AMD Zen 4 Core PMU)
// Event 0x43: LsDmndFillsFromSys (Demand Data Cache Fills by Data Source)
#define ZEN4_EVENT_LS_DMND_FILLS_FROM_SYS 0x43ULL

// UMask 0x02: Local CCX (L3 Cache Hit within the 96 MiB 3D V-Cache CCX)
#define ZEN4_UMASK_L3_HIT_LOCAL_CCX       0x02ULL

// UMask 0x08: DRAM IO Near (L3 Cache Miss served from local DRAM)
#define ZEN4_UMASK_L3_MISS_DRAM_NEAR      0x08ULL

#define ZEN4_RAW_CONFIG_L3_HIT  ((ZEN4_UMASK_L3_HIT_LOCAL_CCX  << 8) | ZEN4_EVENT_LS_DMND_FILLS_FROM_SYS) // 0x0243ULL
#define ZEN4_RAW_CONFIG_L3_MISS ((ZEN4_UMASK_L3_MISS_DRAM_NEAR << 8) | ZEN4_EVENT_LS_DMND_FILLS_FROM_SYS) // 0x0843ULL

static inline long perf_event_open_syscall(struct perf_event_attr* hw_event,
                                           pid_t pid,
                                           int cpu,
                                           int group_fd,
                                           unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int l3_perf_watchdog_open(int* fd_access_out, int* fd_misses_out) {
    if (!fd_access_out || !fd_misses_out) {
        return -1;
    }

    *fd_access_out = -1;
    *fd_misses_out = -1;

    // 1. Configure counter for Local CCX L3 Hits (0x0243)
    struct perf_event_attr pe_hit;
    memset(&pe_hit, 0, sizeof(struct perf_event_attr));
    pe_hit.type           = PERF_TYPE_RAW;
    pe_hit.size           = sizeof(struct perf_event_attr);
    pe_hit.config         = ZEN4_RAW_CONFIG_L3_HIT;
    pe_hit.disabled       = 0;
    pe_hit.exclude_kernel = 1;
    pe_hit.exclude_hv     = 1;

    int fd_hit = (int)perf_event_open_syscall(&pe_hit, 0, -1, -1, 0);
    if (fd_hit < 0) {
        return -1;
    }

    // 2. Configure counter for Local DRAM Fills / L3 Misses (0x0843)
    struct perf_event_attr pe_miss;
    memset(&pe_miss, 0, sizeof(struct perf_event_attr));
    pe_miss.type           = PERF_TYPE_RAW;
    pe_miss.size           = sizeof(struct perf_event_attr);
    pe_miss.config         = ZEN4_RAW_CONFIG_L3_MISS;
    pe_miss.disabled       = 0;
    pe_miss.exclude_kernel = 1;
    pe_miss.exclude_hv     = 1;

    int fd_miss = (int)perf_event_open_syscall(&pe_miss, 0, -1, -1, 0);
    if (fd_miss < 0) {
        close(fd_hit);
        return -1;
    }

    // Reset and enable both counters to guarantee a clean measurement window
    ioctl(fd_hit, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_miss, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_hit, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_miss, PERF_EVENT_IOC_ENABLE, 0);

    *fd_access_out = fd_hit;
    *fd_misses_out = fd_miss;
    return 0;
}

int l3_perf_watchdog_read(int fd_access, int fd_misses, uint64_t* access_out, uint64_t* misses_out) {
    if (fd_access < 0 || fd_misses < 0 || !access_out || !misses_out) {
        return -1;
    }

    uint64_t l3_hits    = 0;
    uint64_t dram_fills = 0;

    ssize_t bytes_hit = read(fd_access, &l3_hits, sizeof(uint64_t));
    if (bytes_hit != (ssize_t)sizeof(uint64_t)) {
        return -1;
    }

    ssize_t bytes_miss = read(fd_misses, &dram_fills, sizeof(uint64_t));
    if (bytes_miss != (ssize_t)sizeof(uint64_t)) {
        return -1;
    }

    // Total L3 demand accesses = L3 hits (Local CCX) + L3 misses (DRAM fills)
    *access_out = l3_hits + dram_fills;
    *misses_out = dram_fills;
    return 0;
}

void l3_perf_watchdog_close(int fd_access, int fd_misses) {
    if (fd_access >= 0) {
        close(fd_access);
    }
    if (fd_misses >= 0) {
        close(fd_misses);
    }
}
