"""
AVX-512 L3 Vector Search Engine & Benchmark
Hardware-accelerated binary Hamming scans and signed INT8 rescoring.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
LIB_DIR = PROJECT_ROOT / "kernels"
LIB_PATH = LIB_DIR / "libl3_vector.so"


class L3VectorEngine:
    """Wrapper around libl3_vector.so for AVX-512 VBMI / VPOPCNTDQ acceleration."""

    def __init__(self, lib_path: Path = LIB_PATH):
        self.lib_path = lib_path
        self._ensure_library()
        self._load_library()

    def _ensure_library(self):
        makefile = LIB_DIR / "Makefile"
        needs_build = not self.lib_path.exists()
        if not needs_build:
            lib_mtime = self.lib_path.stat().st_mtime
            for c_src in LIB_DIR.glob("*.c"):
                if c_src.stat().st_mtime > lib_mtime:
                    needs_build = True
                    break
        if needs_build and makefile.exists():
            subprocess.run(["make", "-C", str(LIB_DIR)], check=True, capture_output=True)
        if not self.lib_path.exists():
            raise FileNotFoundError(f"Shared library not found at {self.lib_path}")

    def _load_library(self):
        self.lib = ctypes.CDLL(str(self.lib_path))

        # avx512_hamming_dist_512
        self.lib.avx512_hamming_dist_512.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.avx512_hamming_dist_512.restype = ctypes.c_uint32

        # avx512_batch_hamming_512
        self.lib.avx512_batch_hamming_512.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint16),
        ]
        self.lib.avx512_batch_hamming_512.restype = None

        # avx512_batch_hamming_512_mt
        self.lib.avx512_batch_hamming_512_mt.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_int,
        ]
        self.lib.avx512_batch_hamming_512_mt.restype = None

        # avx512_int8_dot_512
        self.lib.avx512_int8_dot_512.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.avx512_int8_dot_512.restype = ctypes.c_int32

        # avx512_batch_rerank_candidates
        self.lib.avx512_batch_rerank_candidates.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.avx512_batch_rerank_candidates.restype = None

        # vcache_alloc_thp / vcache_free_thp
        self.lib.vcache_alloc_thp.argtypes = [ctypes.c_size_t]
        self.lib.vcache_alloc_thp.restype = ctypes.c_void_p
        self.lib.vcache_free_thp.argtypes = [ctypes.c_void_p]
        self.lib.vcache_free_thp.restype = None

        # avx512_int8_dot_512_vnni
        self.lib.avx512_int8_dot_512_vnni.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
        ]
        self.lib.avx512_int8_dot_512_vnni.restype = ctypes.c_int32

        # avx512_ivf_rabitq_vnni_search
        self.lib.avx512_ivf_rabitq_vnni_search.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.lib.avx512_ivf_rabitq_vnni_search.restype = ctypes.c_int

    def alloc_thp(self, size_bytes: int) -> int:
        ptr = self.lib.vcache_alloc_thp(size_bytes)
        if not ptr:
            raise MemoryError(f"vcache_alloc_thp({size_bytes}) failed")
        return ptr

    def free_thp(self, ptr: int) -> None:
        if ptr:
            self.lib.vcache_free_thp(ctypes.c_void_p(ptr))

    def int8_dot_vnni(self, q_ptr: int, d_ptr: int, doc_bias: int) -> int:
        return int(
            self.lib.avx512_int8_dot_512_vnni(
                ctypes.c_void_p(q_ptr), ctypes.c_void_p(d_ptr), ctypes.c_int32(doc_bias)
            )
        )

    def batch_hamming(
        self, query: bytes, doc_matrix: bytes, num_docs: int, threads: int = 1
    ) -> list[int]:
        assert len(query) == 64, "Query must be 64 bytes (512 bits)"
        assert len(doc_matrix) >= num_docs * 64, "Doc matrix buffer too small"
        distances = (ctypes.c_uint16 * num_docs)()
        if threads > 1:
            self.lib.avx512_batch_hamming_512_mt(
                query, doc_matrix, num_docs, distances, threads
            )
        else:
            self.lib.avx512_batch_hamming_512(query, doc_matrix, num_docs, distances)
        return list(distances)

    def benchmark_scan(
        self, num_vectors: int = 100_000, iterations: int = 50, threads: int = 1
    ) -> dict:
        """Measure exhaustive L3 scan performance."""
        # Create synthetic 64-byte vectors
        raw_matrix = os.urandom(num_vectors * 64)
        query = os.urandom(64)
        out_buf = (ctypes.c_uint16 * num_vectors)()

        def _scan():
            if threads > 1:
                self.lib.avx512_batch_hamming_512_mt(
                    query, raw_matrix, num_vectors, out_buf, threads
                )
            else:
                self.lib.avx512_batch_hamming_512(
                    query, raw_matrix, num_vectors, out_buf
                )

        # Warmup
        _scan()

        durations = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            _scan()
            durations.append(time.perf_counter_ns() - t0)

        durations.sort()
        median_ns = durations[len(durations) // 2]
        median_us = median_ns / 1_000.0

        qps = 1_000_000_000.0 / median_ns
        bytes_scanned = num_vectors * 64
        bandwidth_gb_s = (bytes_scanned / (median_ns / 1e9)) / (1024**3)
        vectors_per_sec = (num_vectors / (median_ns / 1e9))

        return {
            "num_vectors": num_vectors,
            "index_size_mib": round(bytes_scanned / (1024**2), 2),
            "median_latency_us": round(median_us, 2),
            "qps": round(qps, 1),
            "bandwidth_gb_s": round(bandwidth_gb_s, 2),
            "vectors_per_sec": round(vectors_per_sec, 0),
        }


def read_anon_hugepages_kb() -> int:
    """Read current process AnonHugePages (KiB) from /proc/self/smaps_rollup."""
    smaps = Path("/proc/self/smaps_rollup")
    if not smaps.is_file():
        return 0
    for line in smaps.read_text(encoding="utf-8").splitlines():
        if line.startswith("AnonHugePages:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return 0


class TwoTierIVFRaBitQIndex:
    """Two-tier L3-resident RaBitQ (1-bit VPOPCNTDQ + INT8 VNNI) + THP DDR5-NTA IVF index."""

    DIM = 512

    def __init__(self, engine: L3VectorEngine | None = None, seed: int = 42):
        import numpy as np

        self._np = np
        self.engine = engine or L3VectorEngine()
        self.seed = seed
        self._thp_ptrs: list[int] = []
        self._closed = False
        self.num_docs = 0
        self.num_centroids = 0

        # Deterministic orthogonal RaBitQ rotation matrix (512x512)
        rng = np.random.default_rng(seed)
        q_mat, r_mat = np.linalg.qr(rng.standard_normal((self.DIM, self.DIM), dtype=np.float32))
        signs = np.where(np.diag(r_mat) >= 0, 1.0, -1.0).astype(np.float32)
        self.rot_matrix = np.ascontiguousarray(q_mat * signs[None, :])

        # Buffers kept alive for C pointers
        self.centroid_bin = None
        self.centroid_s8 = None
        self.centroid_bias = None
        self.list_offsets = None
        self.doc_s8_packed = None
        self.doc_bias_packed = None
        self.doc_ids_packed = None

    def _alloc_thp_array(self, shape: tuple[int, ...], dtype) -> "np.ndarray":
        np = self._np
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        ptr = self.engine.alloc_thp(max(64, nbytes))
        self._thp_ptrs.append(ptr)
        buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
        return np.ctypeslib.as_array(buf).view(dtype).reshape(shape)

    @staticmethod
    def _quantize_s8(vecs: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        import numpy as np

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        unit = vecs / norms
        # Uniform global scale (~4*sigma for 512-d rotated unit vectors) preserves cosine monotonicity
        scale = 127.0 / 0.18
        q = np.clip(np.rint(unit * scale), -127, 127).astype(np.int8)
        bias = (-128 * q.astype(np.int32).sum(axis=1)).astype(np.int32)
        return np.ascontiguousarray(q), np.ascontiguousarray(bias)

    @staticmethod
    def _quantize_1bit(vecs: "np.ndarray") -> "np.ndarray":
        import numpy as np

        bits = (vecs >= 0.0).astype(np.uint8)
        return np.ascontiguousarray(np.packbits(bits, axis=1, bitorder="little"))

    def build(
        self,
        vectors_f32: "np.ndarray",
        num_centroids: int = 256,
        niter: int = 12,
        batch_size: int = 8192,
    ) -> dict:
        np = self._np
        self.close()
        self._closed = False

        n_docs, dim = vectors_f32.shape
        assert dim == self.DIM, f"Expected dim={self.DIM}, got {dim}"
        k = min(num_centroids, n_docs)
        self.num_docs = n_docs
        self.num_centroids = k

        # Rotate into isotropic RaBitQ space
        rot_docs = np.ascontiguousarray(vectors_f32.astype(np.float32) @ self.rot_matrix)
        norms = np.linalg.norm(rot_docs, axis=1, keepdims=True)
        rot_docs /= np.where(norms > 1e-12, norms, 1.0)

        # Mini-batch / full-batch spherical k-means with empty-cluster recovery
        rng = np.random.default_rng(self.seed + 1)
        init_idx = rng.choice(n_docs, size=k, replace=False)
        centroids = np.ascontiguousarray(rot_docs[init_idx].copy())
        eff_batch = n_docs if n_docs <= 32768 else min(n_docs, max(batch_size, k * 4))

        for _ in range(niter):
            if eff_batch == n_docs:
                sample = rot_docs
            else:
                sample_idx = rng.choice(n_docs, size=eff_batch, replace=False)
                sample = rot_docs[sample_idx]
            sims = sample @ centroids.T
            assign = np.argmax(sims, axis=1)
            new_c = np.zeros_like(centroids)
            counts = np.bincount(assign, minlength=k)
            np.add.at(new_c, assign, sample)
            non_empty = counts > 0
            new_c[non_empty] /= np.maximum(np.linalg.norm(new_c[non_empty], axis=1, keepdims=True), 1e-12)
            centroids[non_empty] = 0.5 * centroids[non_empty] + 0.5 * new_c[non_empty]
            empty_ids = np.flatnonzero(~non_empty)
            if empty_ids.size > 0:
                reseed = rng.choice(sample.shape[0], size=empty_ids.size, replace=False)
                centroids[empty_ids] = sample[reseed]
            cnorm = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids /= np.where(cnorm > 1e-12, cnorm, 1.0)

        # Assign all docs in chunks to avoid giant N x K matrix
        assign_all = np.empty(n_docs, dtype=np.int32)
        chunk = max(1024, min(16384, n_docs))
        for start in range(0, n_docs, chunk):
            end = min(n_docs, start + chunk)
            sims = rot_docs[start:end] @ centroids.T
            assign_all[start:end] = np.argmax(sims, axis=1)

        order = np.argsort(assign_all, kind="stable")
        sorted_assign = assign_all[order]
        counts = np.bincount(sorted_assign, minlength=k)
        offsets = np.zeros(k + 1, dtype=np.int32)
        np.cumsum(counts, out=offsets[1:])

        # Tier-0a (L3 1-bit RaBitQ) + Tier-0b (L3 INT8 centroids)
        c_bin = self._quantize_1bit(centroids)
        c_s8, c_bias = self._quantize_s8(centroids)

        self.centroid_bin = self._alloc_thp_array(c_bin.shape, np.uint8)
        self.centroid_bin[:] = c_bin
        self.centroid_s8 = self._alloc_thp_array(c_s8.shape, np.int8)
        self.centroid_s8[:] = c_s8
        self.centroid_bias = np.ascontiguousarray(c_bias, dtype=np.int32)
        self.list_offsets = np.ascontiguousarray(offsets, dtype=np.int32)

        # Tier-1 (THP-backed packed INT8 posting lists)
        self.doc_s8_packed = self._alloc_thp_array((n_docs, self.DIM), np.int8)
        self.doc_bias_packed = self._alloc_thp_array((n_docs,), np.int32)
        self.doc_ids_packed = self._alloc_thp_array((n_docs,), np.int32)
        self.doc_ids_packed[:] = order.astype(np.int32)

        for start in range(0, n_docs, chunk):
            end = min(n_docs, start + chunk)
            d_s8, d_bias = self._quantize_s8(rot_docs[order[start:end]])
            self.doc_s8_packed[start:end] = d_s8
            self.doc_bias_packed[start:end] = d_bias

        return {
            "num_docs": n_docs,
            "num_centroids": k,
            "tier0_bytes": int(c_bin.nbytes + c_s8.nbytes + c_bias.nbytes + offsets.nbytes),
            "tier1_bytes": int(
                self.doc_s8_packed.nbytes + self.doc_bias_packed.nbytes + self.doc_ids_packed.nbytes
            ),
            "anon_hugepages_kb": read_anon_hugepages_kb(),
        }

    def search(
        self,
        query_f32: "np.ndarray",
        top_k: int = 10,
        top_c_stage1: int = 512,
        nprobe: int = 48,
        use_nta: bool = True,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        np = self._np
        q_rot = (query_f32.astype(np.float32).reshape(1, self.DIM)) @ self.rot_matrix
        q_bin = self._quantize_1bit(q_rot)
        q_s8, _ = self._quantize_s8(q_rot)

        out_ids = np.empty(top_k, dtype=np.int32)
        out_scores = np.empty(top_k, dtype=np.int32)

        found = self.engine.lib.avx512_ivf_rabitq_vnni_search(
            ctypes.c_void_p(q_bin.ctypes.data),
            ctypes.c_void_p(q_s8.ctypes.data),
            ctypes.c_void_p(self.centroid_bin.ctypes.data),
            ctypes.c_void_p(self.centroid_s8.ctypes.data),
            ctypes.c_void_p(self.centroid_bias.ctypes.data),
            ctypes.c_int(self.num_centroids),
            ctypes.c_int(top_c_stage1),
            ctypes.c_int(nprobe),
            ctypes.c_void_p(self.list_offsets.ctypes.data),
            ctypes.c_void_p(self.doc_s8_packed.ctypes.data),
            ctypes.c_void_p(self.doc_bias_packed.ctypes.data),
            ctypes.c_void_p(self.doc_ids_packed.ctypes.data),
            ctypes.c_int(top_k),
            ctypes.c_void_p(out_ids.ctypes.data),
            ctypes.c_void_p(out_scores.ctypes.data),
            ctypes.c_int(1 if use_nta else 0),
        )
        return out_ids[:found], out_scores[:found]

    def close(self) -> None:
        if not self._closed:
            for ptr in self._thp_ptrs:
                self.engine.free_thp(ptr)
            self._thp_ptrs.clear()
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def main():
    print("=" * 70)
    print(" AVX-512 L3 RESIDENT VECTOR SCAN BENCHMARK (1T vs 8T OpenMP)")
    print("=" * 70)
    engine = L3VectorEngine()
    max_threads = min(8, os.cpu_count() or 1)

    for count in [100_000, 500_000, 1_000_000]:
        for t in sorted({1, max_threads}):
            res = engine.benchmark_scan(num_vectors=count, iterations=30, threads=t)
            print(
                f"Index: {count:>9,} vec ({res['index_size_mib']:>5.2f} MiB) | T={t} | "
                f"{res['median_latency_us']:>7.2f} µs | {res['qps']:>9,.1f} QPS | "
                f"{res['bandwidth_gb_s']:>6.2f} GB/s"
            )
        print("-" * 70)


if __name__ == "__main__":
    main()
