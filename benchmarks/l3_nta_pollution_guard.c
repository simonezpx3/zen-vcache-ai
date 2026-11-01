#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <immintrin.h>

/*
 * Benchmark 11: Non-Temporal Cache Bypass (_MM_HINT_NTA & VMOVNTDQA) vs. L3 Cache Pollution
 * Architecture: AMD Ryzen 7 7800X3D (Zen 4 Exclusive L3 Victim Architecture, 96MB 3D V-Cache)
 *
 * Because Zen 4's L3 is a victim cache filled by L2 evictions, streaming or rescoring
 * large cold buffers from DDR5 using standard temporal loads (`VMOVDQU64` / `_MM_HINT_T0`)
 * pushes cold lines through L2 into L3, evicting the hot working set!
 *
 * This benchmark interleaves:
 *  - Hot Working Set (48 MiB cyclic pointer-chasing & SIMD index in 3D V-Cache)
 *  - Cold DDR5 Interrupter (64 MiB cold DRAM scan between hot index queries)
 * Comparing:
 *  1. Baseline (No Cold Scan — pure 48 MiB L3 residency)
 *  2. Temporal Cold Scan (`_mm512_load_si512` + `_MM_HINT_T0`) -> Pollutes L3
 *  3. Non-Temporal Guard (`_mm512_stream_load_si512` + `_MM_HINT_NTA` + `_mm_clflushopt` eviction guard)
 */

#define HOT_SIZE_MIB  48ULL
#define COLD_SIZE_MIB 256ULL
#define SWEEP_MIB     64ULL
#define HOPS_PER_QUERY 200000
#define ROUNDS 25
#define SEED_DNA 0x732641ULL

static inline uint64_t get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

typedef struct __attribute__((aligned(64))) Node {
    struct Node* next;
    uint64_t pad[7];
} Node;

static Node* build_hot_ring(size_t num_nodes) {
    size_t bytes = num_nodes * sizeof(Node);
    Node* nodes = NULL;
    if (posix_memalign((void**)&nodes, 2 * 1024 * 1024, bytes) != 0 || !nodes) {
        perror("posix_memalign");
        exit(1);
    }
    madvise(nodes, bytes, MADV_HUGEPAGE);

    uint32_t* perm = malloc(num_nodes * sizeof(uint32_t));
    for (size_t i = 0; i < num_nodes; i++) perm[i] = (uint32_t)i;

    uint64_t rng = SEED_DNA;
    for (size_t i = num_nodes - 1; i > 0; i--) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        size_t j = (size_t)(rng % (i + 1));
        uint32_t tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }

    for (size_t i = 0; i < num_nodes - 1; i++) {
        nodes[perm[i]].next = &nodes[perm[i + 1]];
        nodes[perm[i]].pad[0] = i;
    }
    nodes[perm[num_nodes - 1]].next = &nodes[perm[0]];
    free(perm);
    return nodes;
}

static uint64_t* alloc_cold_dram(size_t bytes) {
    uint64_t* buf = NULL;
    if (posix_memalign((void**)&buf, 2 * 1024 * 1024, bytes) != 0 || !buf) {
        perror("posix_memalign");
        exit(1);
    }
    madvise(buf, bytes, MADV_HUGEPAGE);
    memset(buf, 0x5A, bytes);
    return buf;
}

// Standard Temporal Cold Scan: Pollutes L2 -> Evicts into L3 -> Destroys Hot L3 Index
static uint64_t cold_scan_temporal(const uint8_t* cold_base, size_t offset, size_t sweep_bytes) {
    const uint8_t* ptr = cold_base + offset;
    __m512i acc = _mm512_setzero_si512();
    for (size_t i = 0; i < sweep_bytes; i += 64) {
        _mm_prefetch((const char*)(ptr + i + 512), _MM_HINT_T0);
        __m512i v = _mm512_load_si512((const __m512i*)(ptr + i));
        acc = _mm512_add_epi64(acc, v);
    }
    return (uint64_t)_mm512_reduce_add_epi64(acc);
}

// Non-Temporal Guarded Cold Scan: _MM_HINT_NTA + VMOVNTDQA + immediate line invalidation
static uint64_t cold_scan_nontemporal_guarded(uint8_t* cold_base, size_t offset, size_t sweep_bytes) {
    uint8_t* ptr = cold_base + offset;
    __m512i acc = _mm512_setzero_si512();
    for (size_t i = 0; i < sweep_bytes; i += 64) {
        _mm_prefetch((const char*)(ptr + i + 512), _MM_HINT_NTA);
        __m512i v = _mm512_stream_load_si512((__m512i*)(ptr + i));
        acc = _mm512_add_epi64(acc, v);
        // Prevent Zen 4 exclusive L2->L3 victim spill of single-use cold DRAM line
        _mm_clflushopt(ptr + i);
    }
    _mm_sfence();
    return (uint64_t)_mm512_reduce_add_epi64(acc);
}

static Node* warm_hot_ring(Node* start, size_t total_nodes) {
    Node* cur = start;
    for (size_t i = 0; i < total_nodes * 2; i++) {
        cur = cur->next;
    }
    return cur;
}

int main(void) {
    const size_t hot_nodes = (HOT_SIZE_MIB * 1024ULL * 1024ULL) / sizeof(Node);
    const size_t cold_bytes = COLD_SIZE_MIB * 1024ULL * 1024ULL;
    const size_t sweep_bytes = SWEEP_MIB * 1024ULL * 1024ULL;

    printf("====================================================================================================\n");
    printf(" Benchmark 11: Non-Temporal L3 Pollution Guard (48 MiB Hot L3 Index + 64 MiB Cold DDR5 Stream)\n");
    printf("====================================================================================================\n");

    Node* hot_ring = build_hot_ring(hot_nodes);
    uint8_t* cold_dram = (uint8_t*)alloc_cold_dram(cold_bytes);
    volatile uint64_t sink = 0;

    // 1. Pure Hot L3 Baseline (No cold DDR5 interference)
    Node* cur = warm_hot_ring(hot_ring, hot_nodes);
    uint64_t t0 = get_ns();
    for (int r = 0; r < ROUNDS; r++) {
        for (int h = 0; h < HOPS_PER_QUERY; h++) {
            cur = cur->next;
        }
    }
    double base_ns_per_hop = (double)(get_ns() - t0) / ((double)ROUNDS * HOPS_PER_QUERY);
    sink += (uintptr_t)cur;

    // 2. Unprotected Temporal Cold Scan between Hot Queries
    cur = warm_hot_ring(hot_ring, hot_nodes);
    uint64_t hot_temporal_ns = 0;
    for (int r = 0; r < ROUNDS; r++) {
        size_t off = ((size_t)r * sweep_bytes) % (cold_bytes - sweep_bytes);
        sink += cold_scan_temporal(cold_dram, off, sweep_bytes);

        uint64_t q0 = get_ns();
        for (int h = 0; h < HOPS_PER_QUERY; h++) {
            cur = cur->next;
        }
        hot_temporal_ns += (get_ns() - q0);
    }
    double temporal_ns_per_hop = (double)hot_temporal_ns / ((double)ROUNDS * HOPS_PER_QUERY);
    sink += (uintptr_t)cur;

    // 3. Non-Temporal Guarded Cold Scan (_MM_HINT_NTA + VMOVNTDQA + CLFLUSHOPT)
    cur = warm_hot_ring(hot_ring, hot_nodes);
    uint64_t hot_nta_ns = 0;
    for (int r = 0; r < ROUNDS; r++) {
        size_t off = ((size_t)r * sweep_bytes) % (cold_bytes - sweep_bytes);
        sink += cold_scan_nontemporal_guarded(cold_dram, off, sweep_bytes);

        uint64_t q0 = get_ns();
        for (int h = 0; h < HOPS_PER_QUERY; h++) {
            cur = cur->next;
        }
        hot_nta_ns += (get_ns() - q0);
    }
    double nta_ns_per_hop = (double)hot_nta_ns / ((double)ROUNDS * HOPS_PER_QUERY);
    sink += (uintptr_t)cur;

    printf("  1. Quiet Hot 48 MiB L3 Index Baseline            : %6.2f ns / hop (100%% L3 SRAM Hit)\n", base_ns_per_hop);
    printf("  2. After 64 MiB Temporal Cold DDR5 Scan          : %6.2f ns / hop (+%.1f%% L3 Eviction Penalty)\n",
           temporal_ns_per_hop, ((temporal_ns_per_hop / base_ns_per_hop) - 1.0) * 100.0);
    printf("  3. After 64 MiB Non-Temporal Guarded DDR5 Scan   : %6.2f ns / hop (%.2fx faster than Temporal)\n",
           nta_ns_per_hop, temporal_ns_per_hop / nta_ns_per_hop);
    printf("----------------------------------------------------------------------------------------------------\n");
    printf("  Result: Non-Temporal Guard eliminates %.1f%% of the L3 cache eviction latency penalty (sink=%lu)\n",
           ((temporal_ns_per_hop - nta_ns_per_hop) / (temporal_ns_per_hop - base_ns_per_hop)) * 100.0,
           (unsigned long)sink & 0xFF);
    printf("====================================================================================================\n");

    free(cold_dram);
    free(hot_ring);
    return 0;
}
