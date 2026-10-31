#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/*
 * Production AVX-512 Two-Tier L3 HNSW & Vector Search Kernel for Ryzen 7 7800X3D
 *
 * Build shared library:
 *   gcc -O3 -march=znver4 -shared -fPIC examples/l3_vector_kernel.c -o examples/libl3_vector.so
 *
 * Tier 1 (L3 SRAM Resident):
 *   - 512-bit (64-byte) 1-bit Binary Embeddings (AVX-512 VPOPCNTDQ)
 *   - HNSW Layer-0 Navigable Small-World Adjacency Graph (M=16 neighbors/node)
 *   - Upper-Layer Medoid Entry Points
 * Tier 2 (DRAM Rescore):
 *   - 512-dim Signed INT8 Dot-Product Reranking using AVX-512BW
 */

#define HNSW_SEED 0x732641

uint32_t avx512_hamming_dist_512(const uint8_t* a, const uint8_t* b) {
    __m512i va = _mm512_loadu_si512((const __m512i*)a);
    __m512i vb = _mm512_loadu_si512((const __m512i*)b);
    __m512i vx = _mm512_xor_si512(va, vb);
    __m512i vcnt = _mm512_popcnt_epi64(vx);
    return (uint32_t)_mm512_reduce_add_epi64(vcnt);
}

void avx512_batch_hamming_512(
    const uint8_t* query,
    const uint8_t* doc_matrix,
    int num_docs,
    uint16_t* distances_out
) {
    __m512i vq = _mm512_loadu_si512((const __m512i*)query);
    for (int i = 0; i < num_docs; i++) {
        __m512i vd = _mm512_loadu_si512((const __m512i*)(doc_matrix + (size_t)i * 64));
        __m512i vx = _mm512_xor_si512(vq, vd);
        __m512i vcnt = _mm512_popcnt_epi64(vx);
        distances_out[i] = (uint16_t)_mm512_reduce_add_epi64(vcnt);
    }
}

int32_t avx512_int8_dot_512(const int8_t* a, const int8_t* b) {
    __m512i acc = _mm512_setzero_si512();
    for (int i = 0; i < 512; i += 64) {
        __m512i va = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i vb = _mm512_loadu_si512((const __m512i*)(b + i));

        __m512i va_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(va));
        __m512i vb_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(vb));
        __m512i prod_lo = _mm512_madd_epi16(va_lo, vb_lo);

        __m512i va_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(va, 1));
        __m512i vb_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vb, 1));
        __m512i prod_hi = _mm512_madd_epi16(va_hi, vb_hi);

        acc = _mm512_add_epi32(acc, _mm512_add_epi32(prod_lo, prod_hi));
    }
    return _mm512_reduce_add_epi32(acc);
}

void avx512_batch_rerank_candidates(
    const int8_t* query_s8,
    const int8_t* doc_int8_matrix,
    const int32_t* candidate_indices,
    int num_candidates,
    int32_t* scores_out
) {
    for (int i = 0; i < num_candidates; i++) {
        int idx = candidate_indices[i];
        const int8_t* doc_vec = doc_int8_matrix + (size_t)idx * 512;
        scores_out[i] = avx512_int8_dot_512(query_s8, doc_vec);
    }
}

void avx512_hnsw_build_graph(
    const int8_t* doc_int8_matrix,
    int num_docs,
    int degree,
    int32_t* neighbors_out
) {
    if (num_docs <= 0 || degree <= 0) return;
    int near_k = (degree > 4) ? (degree - 4) : degree;
    int long_k = degree - near_k;

    for (int i = 0; i < num_docs; i++) {
        const int8_t* vi = doc_int8_matrix + (size_t)i * 512;
        int32_t best_idx[32];
        int32_t best_sc[32];
        for (int m = 0; m < near_k; m++) {
            best_idx[m] = (i + m + 1) % num_docs;
            best_sc[m] = -2147483647;
        }

        for (int j = 0; j < num_docs; j++) {
            if (i == j) continue;
            int32_t sc = avx512_int8_dot_512(vi, doc_int8_matrix + (size_t)j * 512);
            if (sc > best_sc[near_k - 1]) {
                int pos = near_k - 1;
                while (pos > 0 && sc > best_sc[pos - 1]) {
                    best_sc[pos] = best_sc[pos - 1];
                    best_idx[pos] = best_idx[pos - 1];
                    pos--;
                }
                best_sc[pos] = sc;
                best_idx[pos] = j;
            }
        }

        int32_t* row_out = neighbors_out + (size_t)i * degree;
        for (int m = 0; m < near_k; m++) {
            row_out[m] = best_idx[m];
        }
        uint32_t stride = (uint32_t)(num_docs / (long_k + 1));
        if (stride < 1) stride = 1;
        for (int m = 0; m < long_k; m++) {
            uint32_t hop = ((uint32_t)i + stride * (uint32_t)(m + 1) + (HNSW_SEED & 0x1F)) % (uint32_t)num_docs;
            if ((int)hop == i) hop = (hop + 1) % (uint32_t)num_docs;
            row_out[near_k + m] = (int32_t)hop;
        }
    }
}

int avx512_hnsw_search_tier1(
    const uint8_t* query_bin,
    const uint8_t* doc_bin_matrix,
    const int32_t* hnsw_neighbors,
    int num_docs,
    int degree,
    const int32_t* entry_points,
    int num_entries,
    int max_candidates,
    int32_t* candidate_indices_out
) {
    if (num_docs <= 0 || max_candidates <= 0) return 0;
    if (max_candidates > num_docs) max_candidates = num_docs;

    uint8_t* visited = (uint8_t*)calloc((size_t)num_docs, sizeof(uint8_t));
    int32_t* cand_idx = (int32_t*)malloc((size_t)num_docs * sizeof(int32_t));
    uint16_t* cand_dist = (uint16_t*)malloc((size_t)num_docs * sizeof(uint16_t));
    int32_t* frontier = (int32_t*)malloc((size_t)num_docs * sizeof(int32_t));
    if (!visited || !cand_idx || !cand_dist || !frontier) {
        free(visited); free(cand_idx); free(cand_dist); free(frontier);
        return 0;
    }

    int n_cand = 0;
    int f_head = 0, f_tail = 0;

    for (int e = 0; e < num_entries; e++) {
        int u = entry_points[e];
        if (u >= 0 && u < num_docs && !visited[u]) {
            visited[u] = 1;
            uint16_t d = (uint16_t)avx512_hamming_dist_512(query_bin, doc_bin_matrix + (size_t)u * 64);
            cand_idx[n_cand] = u;
            cand_dist[n_cand] = d;
            n_cand++;
            frontier[f_tail++] = u;
        }
    }

    int max_visits = max_candidates * 2;
    if (max_visits > num_docs) max_visits = num_docs;
    if (max_visits < 128 && num_docs >= 128) max_visits = 128;

    while (f_head < f_tail && n_cand < max_visits) {
        int best_f = f_head;
        uint16_t best_d = 65535;
        int win_end = (f_head + 16 < f_tail) ? (f_head + 16) : f_tail;
        for (int fi = f_head; fi < win_end; fi++) {
            int u = frontier[fi];
            uint16_t d = (uint16_t)avx512_hamming_dist_512(query_bin, doc_bin_matrix + (size_t)u * 64);
            if (d < best_d) {
                best_d = d;
                best_f = fi;
            }
        }
        int cur = frontier[best_f];
        frontier[best_f] = frontier[f_head];
        f_head++;

        const int32_t* nbrs = hnsw_neighbors + (size_t)cur * degree;
        for (int m = 0; m < degree; m++) {
            int nb = nbrs[m];
            if (nb >= 0 && nb < num_docs) {
                _mm_prefetch((const char*)(doc_bin_matrix + (size_t)nb * 64), _MM_HINT_T0);
            }
        }

        for (int m = 0; m < degree; m++) {
            int nb = nbrs[m];
            if (nb < 0 || nb >= num_docs || visited[nb]) continue;
            visited[nb] = 1;
            uint16_t d = (uint16_t)avx512_hamming_dist_512(query_bin, doc_bin_matrix + (size_t)nb * 64);
            cand_idx[n_cand] = nb;
            cand_dist[n_cand] = d;
            n_cand++;
            frontier[f_tail++] = nb;
            if (n_cand >= max_visits) break;
        }
    }

    int out_k = (n_cand < max_candidates) ? n_cand : max_candidates;
    for (int i = 0; i < out_k; i++) {
        int best = i;
        for (int j = i + 1; j < n_cand; j++) {
            if (cand_dist[j] < cand_dist[best]) {
                best = j;
            }
        }
        if (best != i) {
            uint16_t td = cand_dist[i]; cand_dist[i] = cand_dist[best]; cand_dist[best] = td;
            int32_t ti = cand_idx[i]; cand_idx[i] = cand_idx[best]; cand_idx[best] = ti;
        }
        candidate_indices_out[i] = cand_idx[i];
    }

    free(visited);
    free(cand_idx);
    free(cand_dist);
    free(frontier);
    return out_k;
}
