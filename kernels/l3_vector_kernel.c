#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/*
 * AVX-512 Two-Tier L3 HNSW & Vector Search Kernel for Ryzen 7 7800X3D (Zen 4 3D V-Cache)
 *
 * Tier 1 (L3 SRAM Resident):
 *   - 512-bit (64-byte) 1-bit Binary Embeddings (AVX-512 VPOPCNTDQ)
 *   - HNSW Layer-0 Navigable Small-World Adjacency Graph (M=16 neighbors/node)
 *   - Upper-Layer Medoid Entry Points
 * Tier 2 (DRAM Rescore):
 *   - 512-dim Signed INT8 Dot-Product Reranking using AVX-512BW
 */

#define HNSW_SIG 0x732641

// 1. Hamming Distance for 512-bit vector
uint32_t avx512_hamming_dist_512(const uint8_t* a, const uint8_t* b) {
    __m512i va = _mm512_loadu_si512((const __m512i*)a);
    __m512i vb = _mm512_loadu_si512((const __m512i*)b);
    __m512i vx = _mm512_xor_si512(va, vb);
    __m512i vcnt = _mm512_popcnt_epi64(vx);
    return (uint32_t)_mm512_reduce_add_epi64(vcnt);
}

// Batch Hamming scan: 1 query vs num_docs (64 bytes each)
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

// Multi-threaded OpenMP Batch Hamming scan across L3 cores
void avx512_batch_hamming_512_mt(
    const uint8_t* query,
    const uint8_t* doc_matrix,
    int num_docs,
    uint16_t* distances_out,
    int num_threads
) {
    if (num_threads <= 1) {
        avx512_batch_hamming_512(query, doc_matrix, num_docs, distances_out);
        return;
    }
    __m512i vq = _mm512_loadu_si512((const __m512i*)query);
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < num_docs; i++) {
            __m512i vd = _mm512_loadu_si512((const __m512i*)(doc_matrix + (size_t)i * 64));
            __m512i vx = _mm512_xor_si512(vq, vd);
            __m512i vcnt = _mm512_popcnt_epi64(vx);
            distances_out[i] = (uint16_t)_mm512_reduce_add_epi64(vcnt);
        }
    }
}

// 2. Signed INT8 Dot-Product for 512 dimensions (AVX-512BW)
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

// Batch candidate rerank: 1 query int8 vs k candidates from doc matrix
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

// 3. Build L3 HNSW Layer-0 Navigable Small-World Graph (M = degree, default 16)
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
        // Deterministic small-world long-range express links
        uint32_t stride = (uint32_t)(num_docs / (long_k + 1));
        if (stride < 1) stride = 1;
        for (int m = 0; m < long_k; m++) {
            uint32_t hop = ((uint32_t)i + stride * (uint32_t)(m + 1) + (HNSW_SIG & 0x1F)) % (uint32_t)num_docs;
            if ((int)hop == i) hop = (hop + 1) % (uint32_t)num_docs;
            row_out[near_k + m] = (int32_t)hop;
        }
    }
}

// 4. Two-Tier L3 HNSW Graph Traversal (Layer-1 Medoids -> Layer-0 Small-World Walk)
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

    // Seed frontier from upper-layer entry points
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

    // Traverse Layer-0 small-world graph in L3 cache with hardware prefetching
    int max_visits = max_candidates * 2;
    if (max_visits > num_docs) max_visits = num_docs;
    if (max_visits < 128 && num_docs >= 128) max_visits = 128;

    while (f_head < f_tail && n_cand < max_visits) {
        // Pick closest unexpanded node in active frontier window
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

    // Partial selection sort for top max_candidates by Hamming distance
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

// ============================================================================
// v2.0 Two-Tier L3-RaBitQ (VPOPCNTDQ) + INT8 VNNI (VPDPBUSD) + THP Engine
// ============================================================================

void* vcache_alloc_thp(size_t size_bytes) {
    if (size_bytes == 0) {
        size_bytes = 64;
    }
    const size_t two_mib = 2ULL * 1024ULL * 1024ULL;
    size_t align = (size_bytes >= two_mib) ? two_mib : 64ULL;
    size_t alloc_size = ((size_bytes + align - 1ULL) / align) * align;

    void* ptr = aligned_alloc(align, alloc_size);
    if (!ptr && align > 64ULL) {
        alloc_size = ((size_bytes + 63ULL) / 64ULL) * 64ULL;
        ptr = aligned_alloc(64ULL, alloc_size);
    }
    if (ptr) {
#ifdef MADV_HUGEPAGE
        if (alloc_size >= two_mib) {
            madvise(ptr, alloc_size, MADV_HUGEPAGE);
        }
#endif
        memset(ptr, 0, size_bytes);
    }
    return ptr;
}

void vcache_free_thp(void* ptr) {
    free(ptr);
}

static inline void vcache_s8_to_u8_512(const int8_t* src_s8, uint8_t* dst_u8) {
    const __m512i vbias = _mm512_set1_epi8((char)0x80);
    for (int i = 0; i < 512; i += 64) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(src_s8 + i));
        _mm512_store_si512((__m512i*)(dst_u8 + i), _mm512_xor_si512(v, vbias));
    }
}

static inline int32_t vcache_vnni_dot_u8s8(const uint8_t* q_u8, const int8_t* d_s8) {
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    for (int i = 0; i < 512; i += 128) {
        __m512i vq0 = _mm512_load_si512((const __m512i*)(q_u8 + i));
        __m512i vd0 = _mm512_loadu_si512((const __m512i*)(d_s8 + i));
        acc0 = _mm512_dpbusd_epi32(acc0, vq0, vd0);

        __m512i vq1 = _mm512_load_si512((const __m512i*)(q_u8 + i + 64));
        __m512i vd1 = _mm512_loadu_si512((const __m512i*)(d_s8 + i + 64));
        acc1 = _mm512_dpbusd_epi32(acc1, vq1, vd1);
    }
    return _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
}

int32_t avx512_int8_dot_512_vnni(const int8_t* q_s8, const int8_t* d_s8, int32_t doc_bias) {
    uint8_t q_u8[512] __attribute__((aligned(64)));
    vcache_s8_to_u8_512(q_s8, q_u8);
    return vcache_vnni_dot_u8s8(q_u8, d_s8) + doc_bias;
}

static inline void maxheap_u16_push_replace(
    uint16_t* h_dist, int32_t* h_id, int k, uint16_t d, int32_t id
) {
    if (d >= h_dist[0]) return;
    h_dist[0] = d;
    h_id[0] = id;
    int idx = 0;
    while (1) {
        int l = 2 * idx + 1;
        if (l >= k) break;
        int r = l + 1;
        int lg = (r < k && h_dist[r] > h_dist[l]) ? r : l;
        if (h_dist[idx] >= h_dist[lg]) break;
        uint16_t td = h_dist[idx]; h_dist[idx] = h_dist[lg]; h_dist[lg] = td;
        int32_t ti = h_id[idx]; h_id[idx] = h_id[lg]; h_id[lg] = ti;
        idx = lg;
    }
}

static inline void minheap_i32_push_replace(
    int32_t* h_score, int32_t* h_id, int k, int32_t score, int32_t id
) {
    if (score <= h_score[0]) return;
    h_score[0] = score;
    h_id[0] = id;
    int idx = 0;
    while (1) {
        int l = 2 * idx + 1;
        if (l >= k) break;
        int r = l + 1;
        int sm = (r < k && h_score[r] < h_score[l]) ? r : l;
        if (h_score[idx] <= h_score[sm]) break;
        int32_t ts = h_score[idx]; h_score[idx] = h_score[sm]; h_score[sm] = ts;
        int32_t ti = h_id[idx]; h_id[idx] = h_id[sm]; h_id[sm] = ti;
        idx = sm;
    }
}

static inline void minheap_i32_heapify_down(
    int32_t* h_score, int32_t* h_id, int k, int idx
) {
    while (1) {
        int l = 2 * idx + 1;
        if (l >= k) break;
        int r = l + 1;
        int sm = (r < k && h_score[r] < h_score[l]) ? r : l;
        if (h_score[idx] <= h_score[sm]) break;
        int32_t ts = h_score[idx]; h_score[idx] = h_score[sm]; h_score[sm] = ts;
        int32_t ti = h_id[idx]; h_id[idx] = h_id[sm]; h_id[sm] = ti;
        idx = sm;
    }
}

int avx512_ivf_rabitq_vnni_search(
    const uint8_t* query_bin,
    const int8_t* query_s8,
    const uint8_t* centroid_bin,
    const int8_t* centroid_s8,
    const int32_t* centroid_bias,
    int num_centroids,
    int top_c_stage1,
    int nprobe,
    const int32_t* list_offsets,
    const int8_t* doc_s8_packed,
    const int32_t* doc_bias_packed,
    const int32_t* doc_ids_packed,
    int top_k,
    int32_t* out_ids,
    int32_t* out_scores,
    int use_nta
) {
    if (!query_bin || !query_s8 || !centroid_bin || !centroid_s8 || !list_offsets ||
        !doc_s8_packed || !doc_ids_packed || !out_ids || !out_scores) {
        return 0;
    }
    if (num_centroids <= 0 || top_c_stage1 <= 0 || nprobe <= 0 || top_k <= 0) {
        return 0;
    }
    if (top_c_stage1 > num_centroids) top_c_stage1 = num_centroids;
    if (top_c_stage1 > 2048) top_c_stage1 = 2048;
    if (nprobe > top_c_stage1) nprobe = top_c_stage1;
    if (nprobe > 512) nprobe = 512;
    if (top_k > 256) top_k = 256;

    uint8_t q_u8[512] __attribute__((aligned(64)));
    vcache_s8_to_u8_512(query_s8, q_u8);

    // Tier-0a: 1-bit RaBitQ Hamming scan over centroids (VPOPCNTDQ)
    uint16_t s1_dist[2048];
    int32_t s1_id[2048];
    const __m512i vq_bin = _mm512_loadu_si512((const __m512i*)query_bin);
    for (int i = 0; i < top_c_stage1; i++) {
        __m512i vc = _mm512_loadu_si512((const __m512i*)(centroid_bin + (size_t)i * 64));
        __m512i vcnt = _mm512_popcnt_epi64(_mm512_xor_si512(vq_bin, vc));
        s1_dist[i] = (uint16_t)_mm512_reduce_add_epi64(vcnt);
        s1_id[i] = i;
    }
    for (int i = top_c_stage1 / 2 - 1; i >= 0; i--) {
        int idx = i;
        while (1) {
            int l = 2 * idx + 1;
            if (l >= top_c_stage1) break;
            int r = l + 1;
            int lg = (r < top_c_stage1 && s1_dist[r] > s1_dist[l]) ? r : l;
            if (s1_dist[idx] >= s1_dist[lg]) break;
            uint16_t td = s1_dist[idx]; s1_dist[idx] = s1_dist[lg]; s1_dist[lg] = td;
            int32_t ti = s1_id[idx]; s1_id[idx] = s1_id[lg]; s1_id[lg] = ti;
            idx = lg;
        }
    }
    for (int c = top_c_stage1; c < num_centroids; c++) {
        __m512i vc = _mm512_loadu_si512((const __m512i*)(centroid_bin + (size_t)c * 64));
        __m512i vcnt = _mm512_popcnt_epi64(_mm512_xor_si512(vq_bin, vc));
        uint16_t d = (uint16_t)_mm512_reduce_add_epi64(vcnt);
        maxheap_u16_push_replace(s1_dist, s1_id, top_c_stage1, d, c);
    }

    // Tier-0b: Exact INT8 VNNI rescoring of top_c_stage1 centroids (VPDPBUSD)
    int32_t s2_score[512];
    int32_t s2_id[512];
    for (int i = 0; i < nprobe; i++) {
        int cid = s1_id[i];
        int32_t cb = centroid_bias ? centroid_bias[cid] : 0;
        s2_score[i] = vcache_vnni_dot_u8s8(q_u8, centroid_s8 + (size_t)cid * 512) + cb;
        s2_id[i] = cid;
    }
    for (int i = nprobe / 2 - 1; i >= 0; i--) {
        minheap_i32_heapify_down(s2_score, s2_id, nprobe, i);
    }
    for (int i = nprobe; i < top_c_stage1; i++) {
        int cid = s1_id[i];
        int32_t cb = centroid_bias ? centroid_bias[cid] : 0;
        int32_t sc = vcache_vnni_dot_u8s8(q_u8, centroid_s8 + (size_t)cid * 512) + cb;
        minheap_i32_push_replace(s2_score, s2_id, nprobe, sc, cid);
    }

    // Tier-1: Scan DDR5 posting lists for top nprobe clusters with NTA prefetch
    int32_t tk_score[256];
    int32_t tk_id[256];
    int tk_filled = 0;

    for (int p = 0; p < nprobe; p++) {
        int cid = s2_id[p];
        int32_t start = list_offsets[cid];
        int32_t end = list_offsets[cid + 1];
        for (int32_t i = start; i < end; i++) {
            if (i + 4 < end) {
                const char* pf_ptr = (const char*)(doc_s8_packed + (size_t)(i + 4) * 512);
                if (use_nta) {
                    _mm_prefetch(pf_ptr + 0, _MM_HINT_NTA);
                    _mm_prefetch(pf_ptr + 64, _MM_HINT_NTA);
                    _mm_prefetch(pf_ptr + 128, _MM_HINT_NTA);
                    _mm_prefetch(pf_ptr + 192, _MM_HINT_NTA);
                    _mm_prefetch(pf_ptr + 256, _MM_HINT_NTA);
                    _mm_prefetch(pf_ptr + 320, _MM_HINT_NTA);
                    _mm_prefetch(pf_ptr + 384, _MM_HINT_NTA);
                    _mm_prefetch(pf_ptr + 448, _MM_HINT_NTA);
                } else {
                    _mm_prefetch(pf_ptr + 0, _MM_HINT_T0);
                    _mm_prefetch(pf_ptr + 256, _MM_HINT_T0);
                }
            }
            int32_t db = doc_bias_packed ? doc_bias_packed[i] : 0;
            int32_t sc = vcache_vnni_dot_u8s8(q_u8, doc_s8_packed + (size_t)i * 512) + db;
            int32_t doc_id = doc_ids_packed[i];
            if (tk_filled < top_k) {
                tk_score[tk_filled] = sc;
                tk_id[tk_filled] = doc_id;
                tk_filled++;
                if (tk_filled == top_k) {
                    for (int h = top_k / 2 - 1; h >= 0; h--) {
                        minheap_i32_heapify_down(tk_score, tk_id, top_k, h);
                    }
                }
            } else {
                minheap_i32_push_replace(tk_score, tk_id, top_k, sc, doc_id);
            }
        }
    }

    // Sort descending by score
    for (int i = 0; i < tk_filled; i++) {
        int best = i;
        for (int j = i + 1; j < tk_filled; j++) {
            if (tk_score[j] > tk_score[best]) best = j;
        }
        if (best != i) {
            int32_t ts = tk_score[i]; tk_score[i] = tk_score[best]; tk_score[best] = ts;
            int32_t ti = tk_id[i]; tk_id[i] = tk_id[best]; tk_id[best] = ti;
        }
        out_ids[i] = tk_id[i];
        out_scores[i] = tk_score[i];
    }
    return tk_filled;
}
