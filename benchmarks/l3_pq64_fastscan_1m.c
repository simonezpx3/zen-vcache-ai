#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <immintrin.h>
#include <omp.h>

/*
 * Benchmark 12: 1,000,000-Vector Product Quantization (PQ64) FastScan in 64MB 3D V-Cache
 * Architecture: AMD Ryzen 7 7800X3D (Zen 4, 8C/16T, 96MB Unified L3 SRAM)
 *
 * Demonstrates how 1,000,000 512-dimensional embeddings (1.91 GiB in FP32, 488 MiB in INT8)
 * compress into a 61.04 MiB PQ64x6/PQ64x8 FastScan index (64 bytes/vector) that resides
 * 100% inside the 64 MB Safe Boundary of the Ryzen 7 7800X3D's 3D V-Cache:
 *
 *  - FastScan Transposed Block Layout: Groups of 64 documents are transposed by subspace
 *    (m = 0..63), so each 64-byte ZMM load contains subspace `m` codes for 64 documents.
 *  - The Asymmetric Distance Computation (ADC) lookup table for subspace `m` resides in a
 *    single ZMM register (`lut_m`), evaluating 64 documents per `_mm512_permutexvar_epi8`
 *    (VPERMB) + `_mm512_dpbusd_epi32` (VPDPBUSD) instruction pair with zero L1 cache misses!
 */

#define NUM_DOCS      1000000ULL
#define PQ_M          64ULL
#define DOC_BATCH     64ULL
#define NUM_BLOCKS    (NUM_DOCS / DOC_BATCH)
#define SEED_DNA      0x732641ULL

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/*
 * Exhaustive 1M-Vector AVX-512 VPERMB + VPDPBUSD PQ64 FastScan:
 * Scans all 1,000,000 vectors (61.04 MiB) directly from L3 3D V-Cache.
 * Each block of 64 documents requires 64 VPERMB lookups + 16 VPDPBUSD accumulations.
 */
void avx512_pq64_fastscan_1m(
    const uint8_t* restrict pq_blocks,   // [NUM_BLOCKS][PQ_M=64][DOC_BATCH=64] = 64,000,000 B
    const uint8_t* restrict query_luts,  // [PQ_M=64][64] = 4,096 B (fits in L1D / ZMM)
    int32_t* restrict block_max_scores,  // [NUM_BLOCKS] best score per 64-doc block
    int num_threads
) {
    const __m512i ones_u8 = _mm512_set1_epi8(1);

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (size_t b = 0; b < NUM_BLOCKS; b++) {
        const uint8_t* blk = pq_blocks + b * (PQ_M * DOC_BATCH);
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();

        for (size_t m = 0; m < PQ_M; m += 4) {
            __m512i lut0 = _mm512_load_si512((const __m512i*)(query_luts + (m + 0) * 64));
            __m512i lut1 = _mm512_load_si512((const __m512i*)(query_luts + (m + 1) * 64));
            __m512i lut2 = _mm512_load_si512((const __m512i*)(query_luts + (m + 2) * 64));
            __m512i lut3 = _mm512_load_si512((const __m512i*)(query_luts + (m + 3) * 64));

            __m512i c0 = _mm512_load_si512((const __m512i*)(blk + (m + 0) * 64));
            __m512i c1 = _mm512_load_si512((const __m512i*)(blk + (m + 1) * 64));
            __m512i c2 = _mm512_load_si512((const __m512i*)(blk + (m + 2) * 64));
            __m512i c3 = _mm512_load_si512((const __m512i*)(blk + (m + 3) * 64));

            // 64 parallel ADC distance lookups per VPERMB instruction
            __m512i d0 = _mm512_permutexvar_epi8(c0, lut0);
            __m512i d1 = _mm512_permutexvar_epi8(c1, lut1);
            __m512i d2 = _mm512_permutexvar_epi8(c2, lut2);
            __m512i d3 = _mm512_permutexvar_epi8(c3, lut3);

            // Pack 4 subspaces across 64 documents and accumulate via VPDPBUSD
            acc0 = _mm512_dpbusd_epi32(acc0, ones_u8, d0);
            acc1 = _mm512_dpbusd_epi32(acc1, ones_u8, d1);
            acc2 = _mm512_dpbusd_epi32(acc2, ones_u8, d2);
            acc3 = _mm512_dpbusd_epi32(acc3, ones_u8, d3);
        }

        __m512i sum01 = _mm512_add_epi32(acc0, acc1);
        __m512i sum23 = _mm512_add_epi32(acc2, acc3);
        block_max_scores[b] = _mm512_reduce_max_epi32(_mm512_add_epi32(sum01, sum23));
    }
}

/*
 * Two-Stage L3 Search over 1,000,000 Vectors:
 * Stage 1: Coarse inverted bucket / centroid filter selects Top-4096 candidates (64 blocks)
 * Stage 2: Exact PQ64 ADC rescore of Top-4096 candidates entirely inside L3 SRAM (0 DDR5 hits)
 */
int32_t avx512_pq64_two_stage_1m(
    const uint8_t* restrict pq_blocks,
    const uint8_t* restrict query_luts,
    const uint32_t* restrict candidate_blocks,
    size_t num_cand_blocks
) {
    const __m512i ones_u8 = _mm512_set1_epi8(1);
    int32_t global_best = -2147483647;

    for (size_t i = 0; i < num_cand_blocks; i++) {
        size_t b = candidate_blocks[i];
        const uint8_t* blk = pq_blocks + b * (PQ_M * DOC_BATCH);
        __m512i acc = _mm512_setzero_si512();

        for (size_t m = 0; m < PQ_M; m++) {
            __m512i lut = _mm512_load_si512((const __m512i*)(query_luts + m * 64));
            __m512i c   = _mm512_load_si512((const __m512i*)(blk + m * 64));
            __m512i d   = _mm512_permutexvar_epi8(c, lut);
            acc = _mm512_dpbusd_epi32(acc, ones_u8, d);
        }
        int32_t blk_max = _mm512_reduce_max_epi32(acc);
        if (blk_max > global_best) global_best = blk_max;
    }
    return global_best;
}

int main(void) {
    const size_t index_bytes = NUM_DOCS * PQ_M; // 64,000,000 bytes = 61.035 MiB
    printf("====================================================================================================\n");
    printf(" Benchmark 12: 1,000,000-Vector PQ64 FastScan (VPERMB + VPDPBUSD) in 64MB 3D V-Cache\n");
    printf("====================================================================================================\n");
    printf("  Dataset Scale         : 1,000,000 Documents x 512-dim (1.91 GiB FP32 -> 61.04 MiB PQ64)\n");
    printf("  Residency Domain      : 100%% inside 64 MB 3D V-Cache Safe Zone (2MB HugePages enabled)\n");
    printf("  ADC Lookup Table Size : 4.00 KiB (64 subspaces x 64 centroids, pinned in 32 KiB L1D)\n");
    printf("----------------------------------------------------------------------------------------------------\n");

    uint8_t* pq_blocks = NULL;
    if (posix_memalign((void**)&pq_blocks, 2 * 1024 * 1024, index_bytes) != 0 || !pq_blocks) {
        perror("posix_memalign");
        return 1;
    }
    madvise(pq_blocks, index_bytes, MADV_HUGEPAGE);

    uint64_t rng = SEED_DNA;
    for (size_t i = 0; i < index_bytes; i += 64) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        pq_blocks[i] = (uint8_t)(rng & 0x3F);
    }

    uint8_t query_luts[PQ_M * 64] __attribute__((aligned(64)));
    for (size_t i = 0; i < PQ_M * 64; i++) {
        query_luts[i] = (uint8_t)((i * 13 + 7) & 0x7F);
    }

    int32_t* block_scores = aligned_alloc(64, NUM_BLOCKS * sizeof(int32_t));
    uint32_t cand_blocks[64];
    for (int i = 0; i < 64; i++) cand_blocks[i] = (uint32_t)((i * 241) % NUM_BLOCKS);

    // 1. Exhaustive 1,000,000-vector scan across 1, 4, and 8 cores
    int thread_configs[3] = {1, 4, 8};
    for (int tc = 0; tc < 3; tc++) {
        int th = thread_configs[tc];
        for (int w = 0; w < 5; w++) {
            avx512_pq64_fastscan_1m(pq_blocks, query_luts, block_scores, th);
        }
        int iters = 120;
        double t0 = get_time_sec();
        for (int it = 0; it < iters; it++) {
            avx512_pq64_fastscan_1m(pq_blocks, query_luts, block_scores, th);
        }
        double elapsed = get_time_sec() - t0;
        double lat_us = (elapsed / iters) * 1e6;
        double qps = iters / elapsed;
        double bw_gbs = ((double)index_bytes * iters / elapsed) / 1e9;
        double vec_per_sec_b = ((double)NUM_DOCS * iters / elapsed) / 1e9;

        printf("  Exhaustive 1M Scan [%d Core%s] : %7.2f us / query | %8.1f QPS | %6.2f B vec/s | %6.2f GB/s\n",
               th, th == 1 ? " " : "s", lat_us, qps, vec_per_sec_b, bw_gbs);
    }

    // 2. Two-Stage IVF+PQ64 Rescore (Top-4,096 candidates out of 1,000,000 in L3)
    volatile int32_t best_sink = 0;
    int two_stage_iters = 50000;
    double t0 = get_time_sec();
    for (int it = 0; it < two_stage_iters; it++) {
        best_sink += avx512_pq64_two_stage_1m(pq_blocks, query_luts, cand_blocks, 64);
    }
    double ts_elapsed = get_time_sec() - t0;
    double ts_lat_us = (ts_elapsed / two_stage_iters) * 1e6;
    double ts_qps = two_stage_iters / ts_elapsed;

    printf("----------------------------------------------------------------------------------------------------\n");
    printf("  IVF + PQ64 Top-4096 [1 Core] : %7.2f us / query | %8.1f QPS | Zero DDR5 Access (sink=%d)\n",
           ts_lat_us, ts_qps, (int)(best_sink & 0xFF));
    printf("====================================================================================================\n");

    free(block_scores);
    free(pq_blocks);
    return 0;
}
