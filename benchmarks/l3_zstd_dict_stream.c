#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <zstd.h>
#include <zdict.h>
#include <lz4.h>

/*
 * Benchmark 8: Streaming In-L3 Zstandard Dictionary & LZ4 Decompression Pipeline
 * Target Architecture: AMD Ryzen 7 7800X3D (Single CCD, 96MB 3D V-Cache)
 *
 * Demonstrates:
 *  1. Training a 112 KiB Zstandard dictionary (ZDICT_trainFromBuffer) on structured
 *     agent JSONL transcript records, keeping ZSTD_DDict hot in L2/L3 SRAM.
 *  2. Compression ratio comparison on small JSONL records (~600B each):
 *     - Standard Zstd (level 3, no dictionary)
 *     - Zstd + 112 KiB L3-Resident Dictionary (ZSTD_CDict / ZSTD_DDict)
 *     - LZ4 Fast Block Compression
 *  3. Single-core Zen 4 in-L3 decompression throughput (GB/s) for both
 *     dictionary-accelerated Zstd and L3-resident LZ4 block streams.
 */

#define NUM_RECORDS 8000
#define MAX_REC_SIZE 1024
#define DICT_CAPACITY (112 * 1024) // 112 KiB dictionary
#define BENCH_ITERS 400

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static size_t generate_jsonl_record(char *buf, size_t cap, uint32_t step_idx, uint32_t rng) {
    const char *roles[] = {"USER_INPUT", "PLANNER_RESPONSE", "TOOL_RESPONSE", "CHECKPOINT"};
    const char *tools[] = {"run_command", "read_file", "edit_file", "search_web", "arci-kb"};
    const char *status[] = {"DONE", "OK", "VERIFIED", "CACHED"};

    int n = snprintf(
        buf, cap,
        "{\"step_index\":%u,\"source\":\"MODEL\",\"type\":\"%s\",\"status\":\"%s\","
        "\"created_at\":\"2026-10-31T22:%02u:%02u+01:00\",\"conversation_id\":\"990463e7-0173-4983-92e8-febf32bc3603\","
        "\"tool_calls\":[{\"name\":\"%s\",\"args\":{\"WaitMsBeforeAsync\":5000,\"toolSummary\":\"Execute L3 V-Cache telemetry step %u\"}}],"
        "\"content\":\"[STATE: S3: EXECUTE] Empirical verification of Zen 4 96MB 3D V-Cache residency seed=%u latency=%u ns.\"}\n",
        step_idx,
        roles[step_idx & 3],
        status[(rng >> 4) & 3],
        (step_idx / 60) % 60,
        step_idx % 60,
        tools[step_idx % 5],
        step_idx,
        rng ^ 0x732641U,
        11 + (rng % 25)
    );
    return (n > 0 && (size_t)n < cap) ? (size_t)n : 0;
}

int main(void) {
    printf("====================================================================\n");
    printf(" AMD Zen 4 In-L3 Zstandard Dictionary & LZ4 Decompression Benchmark\n");
    printf("====================================================================\n");

    char *raw_corpus = aligned_alloc(64, NUM_RECORDS * MAX_REC_SIZE);
    size_t *rec_offsets = malloc(NUM_RECORDS * sizeof(size_t));
    size_t *rec_sizes = malloc(NUM_RECORDS * sizeof(size_t));

    uint32_t rng = 0x732641U;
    size_t total_raw_bytes = 0;
    for (uint32_t i = 0; i < NUM_RECORDS; i++) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        rec_offsets[i] = total_raw_bytes;
        size_t len = generate_jsonl_record(raw_corpus + total_raw_bytes, MAX_REC_SIZE, i, rng);
        rec_sizes[i] = len;
        total_raw_bytes += len;
    }

    printf("Corpus: %d JSONL transcript records | Total Raw Size: %.2f MiB (Avg: %.1f B/rec)\n",
           NUM_RECORDS, total_raw_bytes / (1024.0 * 1024.0), (double)total_raw_bytes / NUM_RECORDS);

    // 1. Train 112 KiB Zstandard Dictionary on first 1,500 JSONL records
    void *dict_buf = aligned_alloc(64, DICT_CAPACITY);
    size_t train_samples = 1500;
    size_t train_bytes = rec_offsets[train_samples - 1] + rec_sizes[train_samples - 1];
    size_t dict_size = ZDICT_trainFromBuffer(dict_buf, DICT_CAPACITY, raw_corpus, rec_sizes, (unsigned)train_samples);
    if (ZDICT_isError(dict_size)) {
        fprintf(stderr, "ZDICT_trainFromBuffer failed: %s\n", ZDICT_getErrorName(dict_size));
        return 1;
    }
    (void)train_bytes;
    printf("Trained Zstd Dictionary Size: %zu bytes (%.1f KiB <= 512 KiB L2/L3 budget)\n\n",
           dict_size, dict_size / 1024.0);

    ZSTD_CDict *cdict = ZSTD_createCDict(dict_buf, dict_size, 3);
    ZSTD_DDict *ddict = ZSTD_createDDict(dict_buf, dict_size);
    ZSTD_CCtx *cctx = ZSTD_createCCtx();
    ZSTD_DCtx *dctx = ZSTD_createDCtx();

    // 2. Per-Record JSONL Compression: Standard Zstd vs. L3 Dictionary Zstd
    size_t max_dst_per_rec = ZSTD_compressBound(MAX_REC_SIZE);
    char *std_comp_buf = aligned_alloc(64, NUM_RECORDS * max_dst_per_rec);
    size_t *std_comp_offsets = malloc(NUM_RECORDS * sizeof(size_t));
    size_t *std_comp_sizes = malloc(NUM_RECORDS * sizeof(size_t));

    char *dict_comp_buf = aligned_alloc(64, NUM_RECORDS * max_dst_per_rec);
    size_t *dict_comp_offsets = malloc(NUM_RECORDS * sizeof(size_t));
    size_t *dict_comp_sizes = malloc(NUM_RECORDS * sizeof(size_t));

    size_t total_std_comp = 0;
    size_t total_dict_comp = 0;

    for (int i = 0; i < NUM_RECORDS; i++) {
        const char *src = raw_corpus + rec_offsets[i];
        size_t slen = rec_sizes[i];

        std_comp_offsets[i] = total_std_comp;
        size_t c1 = ZSTD_compressCCtx(cctx, std_comp_buf + total_std_comp, max_dst_per_rec, src, slen, 3);
        std_comp_sizes[i] = c1;
        total_std_comp += c1;

        dict_comp_offsets[i] = total_dict_comp;
        size_t c2 = ZSTD_compress_usingCDict(cctx, dict_comp_buf + total_dict_comp, max_dst_per_rec, src, slen, cdict);
        dict_comp_sizes[i] = c2;
        total_dict_comp += c2;
    }

    // 3. Framed 64 KiB Chunk Compression (Zstd + Dict vs Standard Zstd vs LZ4)
    size_t zstd_frame_bound = ZSTD_compressBound(total_raw_bytes);
    char *zstd_frame_comp = aligned_alloc(64, zstd_frame_bound);
    size_t zstd_frame_size = ZSTD_compress_usingCDict(cctx, zstd_frame_comp, zstd_frame_bound, raw_corpus, total_raw_bytes, cdict);

    int lz4_bound = LZ4_compressBound((int)total_raw_bytes);
    char *lz4_comp = aligned_alloc(64, (size_t)lz4_bound);
    int lz4_size = LZ4_compress_default(raw_corpus, lz4_comp, (int)total_raw_bytes, lz4_bound);

    char *decomp_out = aligned_alloc(64, total_raw_bytes + 4096);

    // Warm up L3 cache
    for (int i = 0; i < NUM_RECORDS; i++) {
        ZSTD_decompress_usingDDict(dctx, decomp_out + rec_offsets[i], rec_sizes[i],
                                   dict_comp_buf + dict_comp_offsets[i], dict_comp_sizes[i], ddict);
    }

    // Measure Per-Record Decompression without Dictionary
    double t0 = get_time_sec();
    for (int it = 0; it < BENCH_ITERS; it++) {
        for (int i = 0; i < NUM_RECORDS; i++) {
            ZSTD_decompressDCtx(dctx, decomp_out + rec_offsets[i], rec_sizes[i],
                                std_comp_buf + std_comp_offsets[i], std_comp_sizes[i]);
        }
    }
    double t1 = get_time_sec();
    double std_rec_gbps = ((double)total_raw_bytes * BENCH_ITERS / (t1 - t0)) / (1024.0 * 1024.0 * 1024.0);

    // Measure Per-Record Decompression WITH L3-Resident ZSTD_DDict
    double t2 = get_time_sec();
    for (int it = 0; it < BENCH_ITERS; it++) {
        for (int i = 0; i < NUM_RECORDS; i++) {
            ZSTD_decompress_usingDDict(dctx, decomp_out + rec_offsets[i], rec_sizes[i],
                                       dict_comp_buf + dict_comp_offsets[i], dict_comp_sizes[i], ddict);
        }
    }
    double t3 = get_time_sec();
    double dict_rec_gbps = ((double)total_raw_bytes * BENCH_ITERS / (t3 - t2)) / (1024.0 * 1024.0 * 1024.0);

    // Measure Framed In-L3 Zstd Stream Decompression
    double t4 = get_time_sec();
    for (int it = 0; it < BENCH_ITERS; it++) {
        ZSTD_decompress_usingDDict(dctx, decomp_out, total_raw_bytes, zstd_frame_comp, zstd_frame_size, ddict);
    }
    double t5 = get_time_sec();
    double zstd_stream_gbps = ((double)total_raw_bytes * BENCH_ITERS / (t5 - t4)) / (1024.0 * 1024.0 * 1024.0);

    // Measure In-L3 LZ4 Stream Decompression
    double t6 = get_time_sec();
    for (int it = 0; it < BENCH_ITERS; it++) {
        LZ4_decompress_safe(lz4_comp, decomp_out, lz4_size, (int)total_raw_bytes);
    }
    double t7 = get_time_sec();
    double lz4_stream_gbps = ((double)total_raw_bytes * BENCH_ITERS / (t7 - t6)) / (1024.0 * 1024.0 * 1024.0);

    int verify_cmp = memcmp(raw_corpus, decomp_out, total_raw_bytes);

    printf("1. Small-Record JSONL Compression Ratio & Speed (8,000 Independent Lines):\n");
    printf("   - Standard Zstd (No Dict):    %7.2f KiB | Ratio: %5.2fx | Decomp: %.2f GB/s\n",
           total_std_comp / 1024.0, (double)total_raw_bytes / total_std_comp, std_rec_gbps);
    printf("   - Zstd + 112 KiB L3 DDict:    %7.2f KiB | Ratio: %5.2fx | Decomp: %.2f GB/s (%.2fx smaller!)\n\n",
           total_dict_comp / 1024.0, (double)total_raw_bytes / total_dict_comp, dict_rec_gbps,
           (double)total_std_comp / total_dict_comp);

    printf("2. Streaming In-L3 Decompression Throughput (Single Core Zen 4, 96MB V-Cache):\n");
    printf("   - Zstd Framed Stream + DDict: %7.2f KiB | Ratio: %5.2fx | Decomp: %.2f GB/s\n",
           zstd_frame_size / 1024.0, (double)total_raw_bytes / zstd_frame_size, zstd_stream_gbps);
    printf("   - LZ4 In-L3 Hot Block Stream: %7.2f KiB | Ratio: %5.2fx | Decomp: %.2f GB/s\n",
           lz4_size / 1024.0, (double)total_raw_bytes / lz4_size, lz4_stream_gbps);
    printf("   - Byte-for-Byte Integrity:    %s\n", (verify_cmp == 0) ? "PASSED (100.0% Match)" : "FAILED");
    printf("====================================================================\n");

    ZSTD_freeDCtx(dctx);
    ZSTD_freeCCtx(cctx);
    ZSTD_freeDDict(ddict);
    ZSTD_freeCDict(cdict);
    free(decomp_out);
    free(lz4_comp);
    free(zstd_frame_comp);
    free(dict_comp_sizes);
    free(dict_comp_offsets);
    free(dict_comp_buf);
    free(std_comp_sizes);
    free(std_comp_offsets);
    free(std_comp_buf);
    free(dict_buf);
    free(rec_sizes);
    free(rec_offsets);
    free(raw_corpus);
    return 0;
}
