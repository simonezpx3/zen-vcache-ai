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
 * Benchmark 13: Hybrid Mixture-of-Experts (MoE) — L3-Pinned Shared Expert + DDR5 Cold Experts
 * Architecture: AMD Ryzen 7 7800X3D (Zen 4, 8C/16T, 96MB Unified L3 SRAM)
 *
 * Models a 24-layer MoE forward pass where:
 *  - Gating Router + Shared Expert + Hot KV/Embedding Tables = 48 MiB (L3-resident target)
 *  - Routed Cold Expert Pool = 512 MiB in DDR5 (64 MiB of sparse cold experts streamed per token,
 *    bringing combined working set to 48 + 64 = 112 MiB, crossing the 96 MB 3D V-Cache boundary!)
 *
 * Evaluates three execution regimes:
 *  1. Naive 8-Core MoE (All 112 MiB per token run on 8 cores):
 *     Hits the 4-Core DDR5 Saturation Law on cold experts and thrashes L3 capacity.
 *  2. 4-Core Monolithic MoE (All 112 MiB per token run on 4 cores):
 *     Avoids 8-core DDR5 controller contention, but underutilizes 8-core L3 SRAM bandwidth
 *     on the 48 MiB Shared Expert.
 *  3. Asymmetric V-Cache Partitioned MoE (8-Core L3 Shared Expert + 4-Core Non-Temporal Cold Experts):
 *     Executes the L3-resident 48 MiB Shared Expert across all 8 cores (~620 GB/s L3 SRAM) and
 *     restricts DDR5-bound Cold Experts to 4 cores (`Cores 0-3`) with `_MM_HINT_NTA` non-temporal
 *     hints to prevent L3 cache pollution.
 */

#define SHARED_MIB      48ULL
#define COLD_POOL_MIB   512ULL
#define ACTIVE_COLD_MIB 64ULL
#define NUM_SLICES      (COLD_POOL_MIB / ACTIVE_COLD_MIB)
#define TOKENS_BENCH    40
#define PRNG_INIT_STATE 7546433ULL

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint8_t* alloc_aligned_mib(size_t mib) {
    size_t bytes = mib * 1024ULL * 1024ULL;
    uint8_t* buf = NULL;
    if (posix_memalign((void**)&buf, 2 * 1024 * 1024, bytes) != 0 || !buf) {
        perror("posix_memalign");
        exit(1);
    }
    madvise(buf, bytes, MADV_HUGEPAGE);
    uint64_t rng = PRNG_INIT_STATE ^ mib;
    for (size_t i = 0; i < bytes; i += 64) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        buf[i] = (uint8_t)(rng & 0x3F);
    }
    return buf;
}

static int32_t run_expert_gemv(
    const uint8_t* restrict weights,
    size_t bytes,
    const uint8_t* restrict lut_buf,
    int active_threads,
    int use_nta
) {
    const __m512i ones_u8 = _mm512_set1_epi8(1);
    const size_t chunk = 4096;
    const size_t num_chunks = bytes / chunk;
    int32_t total = 0;

    #pragma omp parallel num_threads(8) reduction(+:total)
    {
        int tid = omp_get_thread_num();
        if (tid < active_threads) {
            __m512i lut = _mm512_load_si512((const __m512i*)lut_buf);
            size_t per_th = num_chunks / (size_t)active_threads;
            size_t start_c = (size_t)tid * per_th;
            size_t end_c = (tid == active_threads - 1) ? num_chunks : start_c + per_th;

            for (size_t c = start_c; c < end_c; c++) {
                const uint8_t* ptr = weights + c * chunk;
                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                for (size_t i = 0; i < chunk; i += 128) {
                    if (use_nta) {
                        _mm_prefetch((const char*)(ptr + i + 512), _MM_HINT_NTA);
                    }
                    __m512i idx0 = _mm512_load_si512((const __m512i*)(ptr + i));
                    __m512i idx1 = _mm512_load_si512((const __m512i*)(ptr + i + 64));
                    __m512i val0 = _mm512_permutexvar_epi8(idx0, lut);
                    __m512i val1 = _mm512_permutexvar_epi8(idx1, lut);
                    acc0 = _mm512_dpbusd_epi32(acc0, ones_u8, val0);
                    acc1 = _mm512_dpbusd_epi32(acc1, ones_u8, val1);
                }
                total += _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
            }
        }
    }
    return total;
}

int main(void) {
    const size_t shared_bytes = SHARED_MIB * 1024ULL * 1024ULL;
    const size_t active_cold_bytes = ACTIVE_COLD_MIB * 1024ULL * 1024ULL;

    printf("====================================================================================================\n");
    printf(" Benchmark 13: Hybrid MoE Offloading — L3 Shared Expert (48 MiB) + DDR5 Cold Experts (64 MiB/tok)\n");
    printf("====================================================================================================\n");
    printf("  MoE Execution Policy                            | Shared Expert | Cold Experts | Total Speed\n");
    printf("----------------------------------------------------------------------------------------------------\n");

    uint8_t* shared_exp = alloc_aligned_mib(SHARED_MIB);
    uint8_t* cold_pool  = alloc_aligned_mib(COLD_POOL_MIB);
    uint8_t lut_buf[64] __attribute__((aligned(64)));
    for (int i = 0; i < 64; i++) lut_buf[i] = (uint8_t)(i + 1);

    volatile int32_t sink = 0;

    for (int mode = 1; mode <= 3; mode++) {
        int shared_threads = (mode == 2) ? 4 : 8;
        int cold_threads   = (mode == 1) ? 8 : 4;
        int use_nta        = (mode == 3) ? 1 : 0;

        // Warmup thread pool and L3 cache
        for (int w = 0; w < 4; w++) {
            sink += run_expert_gemv(shared_exp, shared_bytes, lut_buf, shared_threads, 0);
        }

        double t_shared = 0.0;
        double t_cold = 0.0;

        for (int tok = 0; tok < TOKENS_BENCH; tok++) {
            size_t exp_idx = (size_t)(tok * 3) % NUM_SLICES;
            const uint8_t* cold_slice = cold_pool + exp_idx * active_cold_bytes;

            double s0 = get_time_sec();
            sink += run_expert_gemv(shared_exp, shared_bytes, lut_buf, shared_threads, 0);
            double s1 = get_time_sec();
            t_shared += (s1 - s0);

            sink += run_expert_gemv(cold_slice, active_cold_bytes, lut_buf, cold_threads, use_nta);
            double s2 = get_time_sec();
            t_cold += (s2 - s1);
        }

        double avg_shared_ms = (t_shared / TOKENS_BENCH) * 1000.0;
        double avg_cold_ms   = (t_cold / TOKENS_BENCH) * 1000.0;
        double total_ms      = avg_shared_ms + avg_cold_ms;
        double tok_s         = 1000.0 / total_ms;

        const char* name = (mode == 1) ? "1. Naive 8-Core MoE (8C Shared + 8C DDR5 Cold)" :
                           (mode == 2) ? "2. Uniform 4-Core MoE (4C Shared + 4C DDR5 Cold)" :
                                         "3. Asymmetric V-Cache MoE (8C L3 + 4C NTA Cold)";

        printf("  %-47s | %7.2f ms    | %7.2f ms   | %6.2f ms (%6.1f tok/s)\n",
               name, avg_shared_ms, avg_cold_ms, total_ms, tok_s);
    }

    printf("====================================================================================================\n");
    free(cold_pool);
    free(shared_exp);
    return (sink == 123456789) ? 1 : 0;
}
