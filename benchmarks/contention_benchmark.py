#!/usr/bin/env python3
"""
3D V-Cache Contention & Isolation Benchmark (Ryzen 7 7800X3D)
Measures p50/p90/p99 tail latency of L3 Vector Search under heavy memory bus pressure.
Compares:
  1. Unpinned / Unisolated (threads migrate across all cores 0-7, colliding with heavy workers).
  2. Isolated via vcache-run policy (pinned to cores 0-3, isolating database from compute).
"""

import multiprocessing as mp
import numpy as np
import os
import sys
import time
import json
from pathlib import Path

# Add knowledge directory to path for L3VectorIndex
sys.path.insert(0, str(Path.home() / ".agents/knowledge"))
from l3_vector_index import L3VectorIndex

STRESS_BUFFER_SIZE_MB = 128  # Exceeds 96MB L3 to force DDR5 memory bus flooding
NUM_STRESS_WORKERS = 4       # Saturate cores 4-7
NUM_QUERIES = 400

SAMPLE_QUERIES = [
    "vcache l3 latence 3d cache amd ryzen 7800x3d",
    "beads tracker issues sync",
    "sqlite wal mmap performance zero-copy",
    "hyprland quickshell bar widget qml",
    "avx512 vnni dot product int8 micro model",
    "zero russian policy security hardening",
    "android adb poco f8 ultra crdt",
    "git secret guard scan commit release"
]


def stress_worker(core_id: int, stop_flag: mp.Event):
    """Generates continuous DDR5 memory bus traffic spilling past L3 cache."""
    try:
        os.sched_setaffinity(0, {core_id})
    except Exception:
        pass

    # Allocate 128MB buffer of 64-bit integers
    size = (STRESS_BUFFER_SIZE_MB * 1024 * 1024) // 8
    buf = np.ones(size, dtype=np.int64)

    stride = 64 // 8  # 64-byte cache line stride
    while not stop_flag.is_set():
        # Sequential strided access thrashing L3 cache lines
        buf[::stride] += 1
        # Random sample read
        _ = buf[::stride * 4].sum()


def run_benchmark(idx: L3VectorIndex, affinity_cores: set[int] | None, desc: str) -> list[float]:
    if affinity_cores is not None:
        os.sched_setaffinity(0, affinity_cores)
    else:
        # Allow all cores (0-7)
        os.sched_setaffinity(0, set(range(8)))

    # Warmup
    for q in SAMPLE_QUERIES[:3]:
        _ = idx.query(q, top_k=10, candidate_k=128)

    latencies_us = []
    for i in range(NUM_QUERIES):
        q = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        t0 = time.perf_counter_ns()
        _ = idx.query(q, top_k=10, candidate_k=128)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)

    return latencies_us


def main():
    print("=" * 70)
    print("AMD Ryzen 7 7800X3D: 3D V-Cache Contention & Isolation Benchmark")
    print("=" * 70)

    idx = L3VectorIndex()
    if idx.num_docs == 0:
        print("Error: L3 Vector Index is empty.")
        sys.exit(1)

    print(f"Index loaded: {idx.num_docs} documents, {idx.hot_l3_footprint_bytes / 1024:.2f} KiB L3 Hot Index.")

    # 1. Baseline: Quiet system (no background thrashing)
    print("\nPhase 1 // Measuring Quiet Baseline (No Contention)...")
    quiet_latencies = run_benchmark(idx, {0, 1, 2, 3}, "Quiet System (Cores 0-3)")

    # 2. Start Stress Workers on Cores 4-7
    print(f"\nPhase 2 // Spawning {NUM_STRESS_WORKERS} Memory Thrashers on Cores 4-7 ({STRESS_BUFFER_SIZE_MB}MB each -> DDR5)...")
    stop_event = mp.Event()
    workers = []
    for cid in range(4, 4 + NUM_STRESS_WORKERS):
        p = mp.Process(target=stress_worker, args=(cid, stop_event))
        p.start()
        workers.append(p)

    time.sleep(1.0)  # Let thrashers saturate cache lines and DRAM bus

    try:
        # Scenario A: Unpinned / Unisolated (Cores 0-7)
        print("Phase 3 // Running Scenario A: Unpinned / Default Scheduling (Cores 0-7)...")
        unpinned_latencies = run_benchmark(idx, None, "Unpinned under load")

        # Scenario B: Isolated via vcache-run policy (Pinned to Cores 0-3)
        print("Phase 4 // Running Scenario B: vcache-run Core Isolation (Cores 0-3)...")
        pinned_latencies = run_benchmark(idx, {0, 1, 2, 3}, "Isolated (Cores 0-3)")

    finally:
        print("\nStopping background stress workers...")
        stop_event.set()
        for p in workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()

    # Calculate statistics
    def calc_stats(lats):
        arr = np.array(lats)
        return {
            "p50_us": float(np.percentile(arr, 50)),
            "p90_us": float(np.percentile(arr, 90)),
            "p95_us": float(np.percentile(arr, 95)),
            "p99_us": float(np.percentile(arr, 99)),
            "max_us": float(np.max(arr)),
            "mean_us": float(np.mean(arr)),
            "std_us": float(np.std(arr))
        }

    s_quiet = calc_stats(quiet_latencies)
    s_unpinned = calc_stats(unpinned_latencies)
    s_pinned = calc_stats(pinned_latencies)

    # Format table
    print("\n" + "=" * 78)
    print(f"{'Metric':<18} | {'Quiet (Cores 0-3)':<16} | {'Unpinned under Load':<20} | {'vcache-run Isolated':<18}")
    print("-" * 78)
    for m in ["p50_us", "p90_us", "p95_us", "p99_us", "max_us", "std_us"]:
        name = m.replace("_us", "").upper()
        if name == "STD": name = "JITTER (STD)"
        val_q = s_quiet[m]
        val_u = s_unpinned[m]
        val_p = s_pinned[m]
        v_q = f"{val_q:.1f} µs"
        v_u = f"{val_u:.1f} µs"
        v_p = f"{val_p:.1f} µs"
        print(f"{name:<18} | {v_q:<16} | {v_u:<20} | {v_p:<18}")
    print("=" * 78)

    p99_diff = s_unpinned["p99_us"] - s_pinned["p99_us"]
    jitter_reduction = (s_unpinned["std_us"] - s_pinned["std_us"]) / s_unpinned["std_us"] * 100.0

    print(f"\nResult: vcache-run isolation reduces p99 tail latency by {p99_diff:.1f} µs")
    print(f"Jitter reduction: {jitter_reduction:.1f}% lower standard deviation.")

    # Save results to JSON
    out_file = Path(__file__).resolve().parent.parent / "results/contention_benchmark_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "hardware": "AMD Ryzen 7 7800X3D (8C/8T, 96MB 3D V-Cache)",
        "workload": "L3 Vector RAG Search (1252 docs)",
        "stress_condition": "4x 128MB DDR5 memory thrashing threads on cores 4-7",
        "num_queries": NUM_QUERIES,
        "quiet": s_quiet,
        "unpinned": s_unpinned,
        "vcache_run_isolated": s_pinned,
        "p99_improvement_us": p99_diff,
        "jitter_reduction_pct": jitter_reduction
    }
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Detailed metrics saved to: results/{out_file.name}")


if __name__ == "__main__":
    main()
