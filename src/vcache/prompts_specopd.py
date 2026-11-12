"""
Diverse non-benchmark prompt suite for On-Policy SpecOPD draft training.
Intentionally excludes vcache-spec-bench evaluation prompts (AgentContext, UserMetrics,
binary_search) so downstream acceptance benchmarks remain strictly out-of-sample.
"""

from __future__ import annotations


def _im_prompt(user_msg: str) -> str:
    """Format a Qwen3 zero-thinking chat prompt matching --reasoning-budget 0."""
    return (
        "<|im_start|>user\n"
        f"{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n"
    )


def build_training_prompts() -> list[str]:
    """Return diverse coding, systems, SQL/JSON, and Czech/English technical prompts."""
    raw_code = [
        "pub struct RingBuffer<T, const N: usize> {\n    buf: [Option<T>; N],\n    head: usize,\n    tail: usize,\n}\nimpl<T, const N: usize> RingBuffer<T, N> {\n    pub fn push(&mut self, item: T) -> Result<(), T> {",
        "use std::collections::HashMap;\npub fn merge_counters(a: &HashMap<String, u64>, b: &HashMap<String, u64>) -> HashMap<String, u64> {",
        "fn quicksort(vec: &mut Vec<i32>, lo: usize, hi: usize) {\n    if lo >= hi { return; }\n    let pivot_idx = partition(vec, lo, hi);",
        "pub enum TaskState {\n    Pending { created_at: u64 },\n    Running { worker_id: u32, started_at: u64 },\n    Completed { exit_code: i32 },\n}\nimpl TaskState {\n    pub fn is_terminal(&self) -> bool {",
        "struct TelemetryFrame {\n    timestamp_ns: u64,\n    l3_hits: u64,\n    l3_misses: u64,\n    ipc: f32,\n}\n// Convert TelemetryFrame to JSON serializer method:\nimpl TelemetryFrame {\n    pub fn to_json(&self) -> String {",
        "from dataclasses import dataclass\nfrom pathlib import Path\n\n@dataclass(frozen=True)\nclass CacheStats:\n    hits: int\n    misses: int\n\n    @property\n    def miss_rate(self) -> float:",
        "def chunked_cosine_topk(queries, corpus, k: int = 10, batch_size: int = 1024):",
        "import sqlite3\n\ndef init_fts_store(db_path: str) -> sqlite3.Connection:\n    conn = sqlite3.connect(db_path)\n    conn.execute(",
        "static inline uint64_t popcnt_avx512(const uint64_t *a, const uint64_t *b, size_t words) {\n    uint64_t total = 0;",
        "#include <immintrin.h>\n#include <stdint.h>\n\nvoid avx512_memset_zero(void *dst, size_t bytes) {",
        "CREATE TABLE inference_runs (\n    run_id INTEGER PRIMARY KEY AUTOINCREMENT,\n    model_name TEXT NOT NULL,\n    tok_per_sec REAL NOT NULL,\n    vram_mib INTEGER NOT NULL\n);\n-- Query top 5 fastest runs per model:",
        "SELECT model_name, AVG(tok_per_sec) AS mean_tps\nFROM inference_runs\nWHERE vram_mib <= 8192\nGROUP BY",
        "#!/usr/bin/env bash\nset -euo pipefail\n# Pin process to CCD cores 2-7 and log exit status\nrun_pinned() {",
        "interface RetryPolicy {\n  maxRetries: number;\n  baseDelayMs: number;\n}\nexport function computeBackoff(policy: RetryPolicy, attempt: number): number {",
        "interface ApiResponse<T> {\n  ok: boolean;\n  data?: T;\n  error?: string;\n}\nexport async function fetchWithRetry<T>(url: string, retries = 3): Promise<ApiResponse<T>> {",
        "// Refactor struct Field into builder pattern:\npub struct ServerConfig {\n    pub host: String,\n    pub port: u16,\n    pub threads: u8,\n    pub ctx_len: u32,\n}\npub struct ServerConfigBuilder {",
        "// Re-emit with Clone, PartialEq and builder methods:\npub struct NodeRecord {\n    id: u64,\n    name: String,\n    active: bool,\n    weight: f32,\n}\nimpl NodeRecord {",
        "def parse_cpu_range(spec: str):\n    cores = list()\n    for part in spec.split(','):",
        "import threading\nclass BoundedQueue:\n    def __init__(self, capacity: int):\n        self._cap = capacity",
        "void transpose_8x8_u32(const uint32_t *src, uint32_t *dst, size_t stride) {",
    ]

    chat_topics = [
        "Explain briefly why Transparent HugePages (2MB THP) reduce dTLB misses on 64 MiB working sets.",
        "Write a Rust function to parse key=value lines into a BTreeMap while ignoring comments starting with #.",
        "Show a concise Python context manager that measures wall-clock time in microseconds using time.perf_counter_ns.",
        "Explain the difference between Reverse KL divergence and Forward KL divergence in knowledge distillation.",
        "Write a minimal C function using pthread_setaffinity_np to pin the current thread to CPU cores 2 through 7.",
        "Generate a JSON schema for a hardware benchmark result with fields model, backend, pp_tps, tg_tps, and vram_mib.",
        "Vysvětli stručně česky, proč se u Hybrid MoE na 8jádrovém CPU vyplatí nechat 2 jádra pro operační systém.",
        "Jaký je rozdíl mezi Q8_0 a Q4_0 kvantizací KV cache v llama.cpp z hlediska VRAM a přesnosti?",
        "Napiš krátkou funkci v Pythonu pro výpočet klouzavého průměru ze seznamu čísel bez externích knihoven.",
        "Napiš bash one-liner, který spočítá počet řádků ve všech .py souborech v aktuálním adresáři.",
        "Implement a thread-safe LRU cache in Python using OrderedDict and a threading.Lock.",
        "Write a SQL query to find duplicate email addresses in a users table and keep only the lowest id.",
        "Explain how speculative decoding preserves the exact output distribution of the target model.",
        "Write a short C++ RAII wrapper around a POSIX file descriptor that closes on destruction.",
        "Show how to compute RMSNorm in PyTorch without using built-in nn.RMSNorm.",
        "Write a TypeScript helper that debounces an async function by N milliseconds.",
        "Napiš stručný systemd user service unit pro spuštění lokálního HTTP serveru na portu 8080.",
        "Write a Rust iterator adapter or function that yields overlapping windows of length 4 from a slice.",
        "Explain what AVX-512 VPOPCNTDQ does and why it accelerates binary Hamming search.",
        "Create a Python argparse CLI skeleton with subcommands train, eval, and export.",
        "Write a C function that checks if a buffer is 64-byte aligned and sums its int32 elements.",
        "Jak změřit L3 cache miss rate pomocí perf stat na Linuxu?",
        "Write a Python function to flatten arbitrarily nested lists of integers iteratively.",
        "Show a compact SQLite schema for storing document chunks and FTS5 full-text search triggers.",
    ]

    prompts: list[str] = []
    prompts.extend(raw_code)
    prompts.extend(_im_prompt(t) for t in chat_topics)
    # Add short continuation variations for code & chat diversity
    extra_continuations = [
        "def cosine_similarity(a, b) -> float:\n    dot = sum(x * y for x, y in zip(a, b))\n",
        "pub fn hex_encode(bytes: &Vec<u8>) -> String {\n",
        "SELECT id, created_at, status FROM jobs WHERE status = 'pending' ORDER BY ",
        "for (size_t i = 0; i < count; i += 8) {\n",
        "try {\n  const res = await fetch(endpoint, { method: 'POST',",
        "class SpeculativeConfig:\n    def __init__(self, draft_k: int = 8, min_p: float = 0.0):",
    ]
    prompts.extend(extra_continuations)
    return prompts
