#!/usr/bin/env python3
"""
SQLite L3 Cache Residency Benchmark
Tests FTS5 search query performance with and without PRAGMA mmap_size on Ryzen 7 7800X3D.
"""

import argparse
import os
import sqlite3
import tempfile
import time

DB_ROWS = 40000
QUERY_COUNT = 500

WORDS = [
    "quantum",
    "neural",
    "compiler",
    "vector",
    "latency",
    "pipeline",
    "bandwidth",
    "architecture",
    "hardware",
    "register",
    "assembly",
    "microcode",
    "silicon",
    "cache",
    "residency",
    "memory",
    "paging",
    "mmap",
    "syscall",
    "kernel",
    "instruction",
    "frequency",
    "parallel",
    "dispatch",
    "branch",
    "predictor",
]


def generate_sentence(seed, length=12):
    state = seed ^ 0x732641
    words = []
    for _ in range(length):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        words.append(WORDS[state % len(WORDS)])
    return " ".join(words)


def build_database(db_path):
    print(f"[*] Building benchmark SQLite database at {db_path} ({DB_ROWS} rows)...")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            "CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, body TEXT);"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE docs_fts USING fts5(title, body, tokenize = 'porter unicode61');"
        )

        batch = []
        for i in range(DB_ROWS):
            title = generate_sentence(i, 4)
            body = generate_sentence(i * 3 + 1, 16)
            batch.append((title, body))

        conn.executemany("INSERT INTO docs(title, body) VALUES(?, ?);", batch)
        conn.execute(
            "INSERT INTO docs_fts(rowid, title, body) SELECT id, title, body FROM docs;"
        )

    conn.close()
    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"[*] Database created: {size_mb:.2f} MB")
    return size_mb


def run_benchmark(db_path, use_mmap=False):
    conn = sqlite3.connect(db_path)
    if use_mmap:
        conn.execute("PRAGMA mmap_size = 67108864;")  # 64 MB
        conn.execute("PRAGMA cache_size = -32768;")  # 32 MB
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA synchronous = NORMAL;")

    # Warmup
    for q in ["quantum", "latency", "pipeline"]:
        cur = conn.execute(
            "SELECT count(*) FROM docs_fts WHERE docs_fts MATCH ?;", (q,)
        )
        cur.fetchone()

    queries = []
    for i in range(QUERY_COUNT):
        w1 = WORDS[i % len(WORDS)]
        w2 = WORDS[(i * 3 + 7) % len(WORDS)]
        queries.append(f"{w1} OR {w2}")

    start_time = time.perf_counter()
    total_hits = 0
    for q in queries:
        cur = conn.execute(
            "SELECT rowid, rank FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT 10;",
            (q,),
        )
        total_hits += len(cur.fetchall())
    elapsed = time.perf_counter() - start_time

    conn.close()
    return elapsed, total_hits


def main():
    parser = argparse.ArgumentParser(description="SQLite L3 MMAP Benchmark")
    parser.add_argument("--mode", choices=["baseline", "mmap", "both"], default="both")
    args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="vcache_bench_")
    db_path = os.path.join(tmp_dir, "bench.db")

    try:
        build_database(db_path)

        if args.mode in ("baseline", "both"):
            elapsed_base, hits1 = run_benchmark(db_path, use_mmap=False)
            qps_base = QUERY_COUNT / elapsed_base
            print(
                f"[+] Baseline (Standard I/O): {elapsed_base * 1000:.2f} ms ({qps_base:.1f} QPS, {hits1} hits)"
            )

        if args.mode in ("mmap", "both"):
            elapsed_mmap, hits2 = run_benchmark(db_path, use_mmap=True)
            qps_mmap = QUERY_COUNT / elapsed_mmap
            print(
                f"[+] L3 Zero-Copy MMAP:       {elapsed_mmap * 1000:.2f} ms ({qps_mmap:.1f} QPS, {hits2} hits)"
            )

        if args.mode == "both":
            speedup = (elapsed_base - elapsed_mmap) / elapsed_base * 100
            print(
                f"[+] Performance delta:        {speedup:+.1f}% execution time reduction"
            )
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(tmp_dir)


if __name__ == "__main__":
    main()
