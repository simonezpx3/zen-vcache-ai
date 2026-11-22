#!/usr/bin/env python3
"""Benchmark Two-Tier L3-RaBitQ (1-bit VPOPCNTDQ) + INT8 VNNI + THP DDR5-NTA RAG."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vcache.vector import TwoTierIVFRaBitQIndex, read_anon_hugepages_kb  # noqa: E402

PMU_EVENTS = [
    "ls_dmnd_fills_from_sys.dram_io_near",
    "ls_sw_pf_dc_fills.dram_io_near",
    "l2_pf_miss_l2_l3.all",
    "ls_dmnd_fills_from_sys.local_ccx",
    "ls_sw_pf_dc_fills.local_ccx",
    "l2_pf_miss_l2_hit_l3.all",
]


def _pin_cores(cores: str) -> None:
    if not cores:
        return
    cpus: set[int] = set()
    for part in cores.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    if cpus:
        os.sched_setaffinity(0, cpus)


def _start_perf(pid: int) -> subprocess.Popen | None:
    if not shutil.which("perf"):
        return None
    cmd = ["perf", "stat", "-x,", "-e", ",".join(PMU_EVENTS), "-p", str(pid)]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.05)
        if proc.poll() is not None:
            return None
        return proc
    except Exception:
        return None


def _stop_perf(proc: subprocess.Popen | None) -> dict:
    if proc is None:
        return {"pmu_available": False}
    proc.send_signal(signal.SIGINT)
    _, stderr = proc.communicate(timeout=5)
    raw: dict[str, int] = {}
    for line in stderr.splitlines():
        cols = line.split(",")
        if len(cols) >= 3:
            val_str, _, ev_name = cols[0].strip(), cols[1], cols[2].strip()
            ev_base = ev_name.split(":")[0]
            if val_str.isdigit():
                raw[ev_base] = int(val_str)
    if not raw:
        return {"pmu_available": False, "pmu_stderr": stderr.strip()[:300]}

    dram_fills = (
        raw.get("ls_dmnd_fills_from_sys.dram_io_near", 0)
        + raw.get("ls_sw_pf_dc_fills.dram_io_near", 0)
        + raw.get("l2_pf_miss_l2_l3.all", 0)
    )
    l3_hits = (
        raw.get("ls_dmnd_fills_from_sys.local_ccx", 0)
        + raw.get("ls_sw_pf_dc_fills.local_ccx", 0)
        + raw.get("l2_pf_miss_l2_hit_l3.all", 0)
    )
    total_l3_resolved = dram_fills + l3_hits
    return {
        "pmu_available": True,
        "events": raw,
        "true_core_dram_bytes": dram_fills * 64,
        "true_core_l3_hit_bytes": l3_hits * 64,
        "l3_hit_rate": round(l3_hits / total_l3_resolved, 4) if total_l3_resolved > 0 else None,
    }


def _build_streaming_10m(
    num_docs: int,
    num_centroids: int,
    num_queries: int,
    seed: int = 42,
) -> tuple[TwoTierIVFRaBitQIndex, np.ndarray, list[set[int]], dict]:
    """Stream-build up to 10M clustered 512-d vectors without holding 20 GiB fp32 in RAM."""
    dim = 512
    rng = np.random.default_rng(seed)
    index = TwoTierIVFRaBitQIndex(seed=seed)

    # Generate isotropic unit cluster centers and rotate into RaBitQ space
    raw_centers = rng.standard_normal((num_centroids, dim), dtype=np.float32)
    raw_centers /= np.linalg.norm(raw_centers, axis=1, keepdims=True)
    rot_centers = np.ascontiguousarray(raw_centers @ index.rot_matrix)
    rot_centers /= np.linalg.norm(rot_centers, axis=1, keepdims=True)

    # Queries sampled near random clusters
    q_cids = rng.integers(0, num_centroids, size=num_queries)
    q_sigma = np.float32(0.35 / np.sqrt(dim))
    queries = raw_centers[q_cids] + q_sigma * rng.standard_normal((num_queries, dim), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    rot_queries = np.ascontiguousarray(queries @ index.rot_matrix)

    # Deterministic balanced cluster sizes summing to num_docs
    base = num_docs // num_centroids
    rem = num_docs % num_centroids
    counts = np.full(num_centroids, base, dtype=np.int32)
    counts[:rem] += 1
    offsets = np.zeros(num_centroids + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])

    c_bin = index._quantize_1bit(rot_centers)
    c_s8, c_bias = index._quantize_s8(rot_centers)
    index.num_docs = num_docs
    index.num_centroids = num_centroids
    index.centroid_bin = index._alloc_thp_array(c_bin.shape, np.uint8)
    index.centroid_bin[:] = c_bin
    index.centroid_s8 = index._alloc_thp_array(c_s8.shape, np.int8)
    index.centroid_s8[:] = c_s8
    index.centroid_bias = np.ascontiguousarray(c_bias, dtype=np.int32)
    index.list_offsets = np.ascontiguousarray(offsets, dtype=np.int32)

    index.doc_s8_packed = index._alloc_thp_array((num_docs, dim), np.int8)
    index.doc_bias_packed = index._alloc_thp_array((num_docs,), np.int32)
    index.doc_ids_packed = index._alloc_thp_array((num_docs,), np.int32)
    index.doc_ids_packed[:] = np.arange(num_docs, dtype=np.int32)

    # Track running exact top-10 ground truth per query while streaming clusters
    gt_scores = np.full((num_queries, 10), -1e9, dtype=np.float32)
    gt_ids = np.full((num_queries, 10), -1, dtype=np.int32)

    doc_sigma = np.float32(0.45 / np.sqrt(dim))
    chunk_clusters = 256
    for c_start in range(0, num_centroids, chunk_clusters):
        c_end = min(num_centroids, c_start + chunk_clusters)
        d_start = int(offsets[c_start])
        d_end = int(offsets[c_end])
        n_chunk = d_end - d_start

        c_rep = np.repeat(rot_centers[c_start:c_end], counts[c_start:c_end], axis=0)
        vecs = c_rep + doc_sigma * rng.standard_normal((n_chunk, dim), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        d_s8, d_bias = index._quantize_s8(vecs)
        index.doc_s8_packed[d_start:d_end] = d_s8
        index.doc_bias_packed[d_start:d_end] = d_bias

        # Exact float32 similarity against queries (rot_queries @ vecs.T is orthogonal-exact)
        sims = rot_queries @ vecs.T  # (Q, n_chunk)
        local_top_idx = np.argpartition(-sims, 10, axis=1)[:, :10]
        local_top_sc = np.take_along_axis(sims, local_top_idx, axis=1)
        local_top_ids = (local_top_idx + d_start).astype(np.int32)

        comb_sc = np.concatenate([gt_scores, local_top_sc], axis=1)
        comb_id = np.concatenate([gt_ids, local_top_ids], axis=1)
        top_ord = np.argpartition(-comb_sc, 10, axis=1)[:, :10]
        gt_scores = np.take_along_axis(comb_sc, top_ord, axis=1)
        gt_ids = np.take_along_axis(comb_id, top_ord, axis=1)

    gt_sets = [set(gt_ids[q].tolist()) for q in range(num_queries)]
    meta = {
        "num_docs": num_docs,
        "num_centroids": num_centroids,
        "tier0_bytes": int(c_bin.nbytes + c_s8.nbytes + c_bias.nbytes + offsets.nbytes),
        "tier1_bytes": int(
            index.doc_s8_packed.nbytes + index.doc_bias_packed.nbytes + index.doc_ids_packed.nbytes
        ),
        "anon_hugepages_kb": read_anon_hugepages_kb(),
    }
    return index, queries, gt_sets, meta


def run_benchmark(
    num_docs: int,
    num_centroids: int,
    top_c_stage1: int,
    nprobe: int,
    num_queries: int,
    iters_per_query: int,
    use_nta: bool,
    cores: str,
) -> dict:
    _pin_cores(cores)
    t_build0 = time.perf_counter()
    index, queries, gt_sets, meta = _build_streaming_10m(
        num_docs=num_docs, num_centroids=num_centroids, num_queries=num_queries
    )
    build_s = time.perf_counter() - t_build0

    try:
        # Warmup L3 Tier-0 tables
        for q in queries[: min(16, num_queries)]:
            index.search(q, top_k=10, top_c_stage1=top_c_stage1, nprobe=nprobe, use_nta=use_nta)

        recalls: list[float] = []
        lat_us: list[float] = []

        perf_proc = _start_perf(os.getpid())
        t0_all = time.perf_counter()
        for r in range(iters_per_query):
            for qi in range(num_queries):
                t0 = time.perf_counter_ns()
                ids, _ = index.search(
                    queries[qi],
                    top_k=10,
                    top_c_stage1=top_c_stage1,
                    nprobe=nprobe,
                    use_nta=use_nta,
                )
                lat_us.append((time.perf_counter_ns() - t0) / 1_000.0)
                if r == 0:
                    recalls.append(len(set(ids.tolist()) & gt_sets[qi]) / 10.0)
        elapsed_s = time.perf_counter() - t0_all
        pmu = _stop_perf(perf_proc)

        lat_arr = np.array(lat_us, dtype=np.float64)
        total_q = len(lat_us)
        if pmu.get("pmu_available"):
            pmu["dram_bytes_per_query"] = round(pmu["true_core_dram_bytes"] / total_q, 1)
            pmu["l3_hit_bytes_per_query"] = round(pmu["true_core_l3_hit_bytes"] / total_q, 1)

        return {
            "mode": "10M" if num_docs >= 10_000_000 else "1M_smoke",
            "cores": cores or "unpinned",
            "use_nta": use_nta,
            "num_docs": num_docs,
            "num_centroids": num_centroids,
            "top_c_stage1": top_c_stage1,
            "nprobe": nprobe,
            "num_queries": total_q,
            "build_seconds": round(build_s, 2),
            "tier0_mib": round(meta["tier0_bytes"] / (1024**2), 2),
            "tier1_gib": round(meta["tier1_bytes"] / (1024**3), 3),
            "anon_hugepages_kb": read_anon_hugepages_kb(),
            "recall_at_10": round(float(np.mean(recalls)), 4),
            "p50_us": round(float(np.percentile(lat_arr, 50)), 2),
            "p95_us": round(float(np.percentile(lat_arr, 95)), 2),
            "p99_us": round(float(np.percentile(lat_arr, 99)), 2),
            "qps": round(total_q / elapsed_s, 1),
            "pmu": pmu,
        }
    finally:
        index.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-Tier L3-RaBitQ + INT8 VNNI Benchmark")
    parser.add_argument("--smoke-1m", action="store_true", help="Run 1M vector smoke benchmark")
    parser.add_argument("--full-10m", action="store_true", help="Run full 10M vector benchmark")
    parser.add_argument("--cores", default="2-7", help="CPU core list for affinity (default: 2-7)")
    parser.add_argument("--no-nta", action="store_true", help="Use _MM_HINT_T0 instead of _MM_HINT_NTA")
    parser.add_argument("--queries", type=int, default=50, help="Unique queries for recall/latency")
    parser.add_argument("--iters", type=int, default=10, help="Query loop repetitions")
    parser.add_argument("--json-out", type=Path, default=None, help="Path to write JSON results")
    args = parser.parse_args()

    if args.full_10m:
        num_docs, num_centroids, top_c, nprobe = 10_000_000, 65_536, 512, 48
    else:
        num_docs, num_centroids, top_c, nprobe = 1_000_000, 8_192, 256, 32

    res = run_benchmark(
        num_docs=num_docs,
        num_centroids=num_centroids,
        top_c_stage1=top_c,
        nprobe=nprobe,
        num_queries=args.queries,
        iters_per_query=args.iters,
        use_nta=not args.no_nta,
        cores=args.cores,
    )
    print(json.dumps(res, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
