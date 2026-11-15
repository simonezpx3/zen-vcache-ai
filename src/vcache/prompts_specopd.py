"""
Diverse non-benchmark prompt suite for On-Policy SpecOPD draft training.
Intentionally excludes vcache-spec-bench evaluation prompts (AgentContext, UserMetrics,
binary_search) so downstream acceptance benchmarks remain strictly out-of-sample.
"""

from __future__ import annotations


def _im_prompt(user_msg: str, style: str = "qwen3") -> str:
    """Format a zero-thinking chat prompt matching Qwen3 or DeepSeek-R1-Qwen3."""
    if style == "deepseek":
        return f"<｜begin▁of▁sentence｜><｜User｜>{user_msg}<｜Assistant｜><think>\n</think>\n"
    return (
        "<|im_start|>user\n"
        f"{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n"
    )


def _expanded_prompts(style: str) -> list[str]:
    """Return ~100 additional non-benchmark prompts for larger-corpus distillation."""
    code_ext = [
        "use std::collections::BTreeMap;\npub fn top_k_counts(items: Vec<String>, k: usize) -> Vec<(String, usize)> {\n",
        "#[derive(Debug, Clone)]\npub struct MetricsSnapshot {\n    pub cpu_pct: f32,\n    pub rss_mib: u64,\n    pub ctx_switches: u64,\n}\nimpl MetricsSnapshot {\n    pub fn delta(&self, prev: &Self) -> Self {\n",
        "pub trait PacketCodec {\n    type Error;\n    fn encode(&self, out: &mut Vec<u8>) -> Result<(), Self::Error>;\n}\npub struct FrameHeader {\n    pub magic: u32,\n    pub seq: u64,\n    pub len: u32,\n}\nimpl PacketCodec for FrameHeader {\n",
        "pub fn sliding_window_max(nums: &Vec<i32>, k: usize) -> Vec<i32> {\n    use std::collections::VecDeque;\n",
        "pub fn parse_env_pairs(input: &str) -> std::collections::HashMap<String, String> {\n",
        "// Refactor struct fields cleanly into a new immutable summary struct:\npub struct WorkerTelemetry {\n    pub worker_id: u32,\n    pub queued_jobs: usize,\n    pub completed_jobs: usize,\n    pub failed_jobs: usize,\n    pub mean_latency_ms: f64,\n}\n// Re-emit matching fields under WorkerSummary:\npub struct WorkerSummary {\n",
        "// Duplicate struct with identical field names for snapshot serialization:\npub struct SessionLedger {\n    pub tenant_key: String,\n    pub allocated_bytes: u64,\n    pub expired_entries: u32,\n    pub last_flush_epoch: u64,\n}\npub struct SessionLedgerSnapshot {\n",
        "import asyncio\nfrom dataclasses import dataclass\n\n@dataclass\nclass RetryConfig:\n    attempts: int = 4\n    delay_s: float = 0.25\n\nasync def run_with_backoff(fn, cfg: RetryConfig):\n",
        "def iter_chunks(seq, size: int):\n    if size <= 0:\n        raise ValueError('size must be positive')\n",
        "def merge_sorted_unique(left, right):\n    i = j = 0\n    out = []\n",
        "import json\nfrom pathlib import Path\n\ndef atomic_write_json(path: Path, payload: dict) -> None:\n",
        "import hashlib\n\ndef content_digest_hex(chunks) -> str:\n    h = hashlib.blake2b(digest_size=32)\n",
        "#include <stdint.h>\n#include <stddef.h>\n\nvoid xor_bytes_inplace(uint8_t *dst, const uint8_t *src, size_t n) {\n",
        "#include <stdlib.h>\n\ntypedef struct Node {\n    int key;\n    struct Node *next;\n} Node;\n\nNode *reverse_list(Node *head) {\n",
        "template <typename T>\nclass RingSpan {\npublic:\n    explicit RingSpan(size_t cap) : cap_(cap), head_(0), tail_(0) {}\n",
        "CREATE TABLE job_events (\n    event_id INTEGER PRIMARY KEY,\n    job_name TEXT NOT NULL,\n    status TEXT NOT NULL,\n    duration_ms REAL NOT NULL,\n    created_at INTEGER NOT NULL\n);\n-- Compute p50 and count per job_name for completed runs:\nSELECT job_name, COUNT(*) AS runs,",
        "WITH recent_runs AS (\n    SELECT model, tokens_per_sec, ROW_NUMBER() OVER (PARTITION BY model ORDER BY id DESC) AS rn\n    FROM bench_history\n)\nSELECT model, AVG(tokens_per_sec) FROM recent_runs WHERE ",
        "export interface PaginatedResult<T> {\n  items: T[];\n  nextCursor: string | null;\n  totalEstimate: number;\n}\nexport function mapPage<A, B>(page: PaginatedResult<A>, fn: (x: A) -> B): PaginatedResult<B> {\n",
        "type Validator<T> = (val: unknown) => T | null;\nexport function parseJsonSafe<T>(raw: string, check: Validator<T>): T | null {\n",
        "#!/usr/bin/env bash\nset -euo pipefail\n# Archive directory older than 7 days with zstd\narchive_old_logs() {\n",
        "pub fn levenshtein_distance(a: &str, b: &str) -> usize {\n    let b_bytes = b.as_bytes();\n",
        "pub fn decode_varint(buf: &Vec<u8>, offset: &mut usize) -> Option<u64> {\n",
        "class TokenBucket:\n    def __init__(self, rate_per_sec: float, capacity: float):\n        self.rate = rate_per_sec\n",
        "def topological_sort(edges):\n    from collections import defaultdict, deque\n",
        "def reservoir_sample(stream, k: int, seed: int = 0):\n    import random\n",
        "size_t align_up(size_t value, size_t alignment) {\n",
        "int popcount_u64_slice(const uint64_t *data, size_t len) {\n",
        "// Implement ThreadPool submit returning a channel receiver in Rust:\nuse std::sync::mpsc;\npub fn spawn_worker_pool(workers: usize) -> ",
        "SELECT u.account_id, COALESCE(SUM(e.bytes_in), 0) AS total_bytes\nFROM accounts u\nLEFT JOIN transfer_events e ON u.account_id = e.account_id\nGROUP BY ",
        "function createEmitter() {\n  const listeners = new Map();\n  return {\n",
    ]
    chat_ext = [
        "Write a Rust fn to deduplicate sorted Vec<i32> in-place and return new length.",
        "Write a Rust helper formatting nanoseconds as us, ms, or s.",
        "Explain how memory bandwidth limits autoregressive LLM decode vs compute-bound prefill.",
        "Show a Rust exponential moving average (EMA) struct with update and value methods.",
        "Write a Python generator reading a large file line-by-line in 64 KB chunks.",
        "Explain why tied input/LM-head weights halve parameter count in tiny LMs.",
        "Write a C function computing lookup-free byte-by-byte CRC32C over n bytes.",
        "Show a lock-free SPSC ring buffer index update using C11 atomics.",
        "Write a SQL CTE query finding sessions with event gaps over 300 seconds.",
        "Napiš v Pythonu kosinovou podobnost dvou vektorů bez NumPy s ošetřením nuly.",
        "Vysvětli česky rozdíl mezi L1/L2 cache a 3D V-Cache L3 na Ryzen X3D.",
        "Jak funguje speculative decoding a proč nemění výstupní distribuci cílového modelu?",
        "Napiš v Rustu parser jednoho CSV řádku s uvozovkami bez externích crate.",
        "Write a bash snippet waiting up to 30s for a local TCP port using nc or /dev/tcp.",
        "Show cosine LR decay with linear warmup in pure PyTorch.",
        "Write a TypeScript bounded FIFO cache class with O(1) push and shift via Map.",
        "Explain Q8_0 block quantization in GGML (32 weights per block, one fp16 scale).",
        "Write a Python function validating balanced parentheses, braces, and angle brackets.",
        "Show a minimal Rust error enum implementing Display and std::error::Error.",
        "Write SQLite pragmas optimizing local log write throughput (WAL, synchronous).",
        "Why does acceptance rate matter more than draft speed once draft latency is < 0.5 ms?",
        "Write a Python helper grouping epoch timestamps into 5-minute buckets.",
        "Vysvětli česky kvantizaci KV cache (Q8_0 vs Q4_0) a dopad na dlouhý kontext.",
        "Napiš v C rychlý skalární součin dvou float vektorů s 4x unrollingem smyčky.",
        "Write a Rust flat row-major 2D dense matrix struct with checked get and set.",
        "Show how to inspect Linux hugepages and NUMA topology via /proc and /sys.",
        "Write a Python function merging overlapping (start, end) integer intervals.",
        "Explain Reverse KL intuition in On-Policy KD (mode-seeking vs mode-covering).",
        "Summarize key JSON parser pitfalls: surrogate pairs, escapes, number precision.",
        "Napiš systemd timer + service unit pro denní spuštění lokálního benchmarku.",
        "Write a Rust run-length encoder over bytes returning Vec<(u8, usize)>.",
        "Show a Python online mean/variance tracker using Welford's algorithm.",
        "Explain how RoPE encodes relative token offsets via 2D rotation matrices.",
        "Write a C++ template clamp_slice clamping elements between lo and hi.",
        "Napiš bash funkci pro bezpečné zamčení skriptu přes flock na lock soubor.",
        "Write a SQL query computing 7-day rolling sum using window frames.",
        "Explain how GQA shrinks KV cache size vs standard Multi-Head Attention.",
        "Write a Python helper flattening nested dicts into dot-separated keys.",
        "Show a Rust BufRead helper counting non-empty lines and bytes from stdin.",
        "Vysvětli česky, proč cross-model drafting selže při jediném odlišném speciálním tokenu.",
        "Write a TypeScript async semaphore limiting concurrent promises to maxConcurrency.",
        "Show gradient accumulation with mixed-precision autocast in PyTorch.",
        "Write a C in-place binary search returning index or -1 on sorted int64_t.",
        "Explain why RMSNorm is cheaper than LayerNorm with similar Transformer stability.",
        "Write a Python Levenshtein edit distance function using O(min(m, n)) memory.",
        "Napiš v Rustu minimal CLI parser pro --threads N a --verbose bez clap.",
        "Convert an F32 weight tensor to symmetric per-group 8-bit blocks in NumPy.",
        "Write a SQL query detecting gaps in monotonic integer IDs.",
        "Explain the tradeoff between draft hidden width and CPU L3 cache residency.",
        "Write a Python Sieve of Eratosthenes returning all primes up to N.",
        "Show a minimal C++ RAII scope guard executing a lambda on exit.",
        "Vysvětli stručně, co měří PMU L3 cache miss rate a jak ovlivňuje latenci.",
        "Write a Rust word-frequency counter sorted descending from whitespace tokens.",
        "Show how to time CUDA kernels in PyTorch using torch.cuda.Event.",
        "Write a Python function computing longest common prefix across strings.",
        "Explain SwiGLU gated activation vs standard ReLU MLP layers.",
        "Write a bash find+xargs pipeline summing byte size of all .rs files.",
        "Vysvětli česky rozdíl mezi n-gram spekulací (0 VRAM) a neuronovým draftem.",
        "Write a Rust base64url encoder without padding.",
        "Show top-k and nucleus (top-p) token filtering in PyTorch logits.",
        "Write a Python 3-color DFS cycle detector on a directed adjacency dict.",
        "Explain mmap GGUF cold page faults vs warm L3 cache residency.",
        "Write a C locale-independent case-insensitive string comparator.",
        "Napiš v Pythonu jednoduchý LRU dekorátor přes collections.OrderedDict.",
        "Write a SQLite JSON_EXTRACT query filtering rows by nested status.",
        "Why does greedy temp=0 verification need top-1 agreement over broad entropy?",
        "Write a Rust iterator helper yielding N-element chunks from any Iterator.",
        "Show how to benchmark CPU sequential bandwidth safely in C.",
        "Write a Python snippet comparing two JSONL files by line count and schema.",
        "Vysvětli česky kombinaci CE loss a top-K KL při destilaci draft modelu.",
    ]
    out: list[str] = []
    out.extend(code_ext)
    out.extend(_im_prompt(t, style=style) for t in chat_ext)
    return out


def build_training_prompts(expanded: bool = False, style: str = "qwen3") -> list[str]:
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
    prompts.extend(_im_prompt(t, style=style) for t in chat_topics)
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
    if expanded:
        prompts.extend(_expanded_prompts(style=style))
    return prompts
