"""
Zen 4 3D V-Cache Speculative Decoding Benchmark Harness
Comprehensive empirical comparison of Baseline, Unpinned SMT, and L3-Pinned modes.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .topology import get_cpu_topology


def _resolve_default_llama_cli() -> Path:
    found = shutil.which("llama-cli")
    if found:
        return Path(found)
    return Path.home() / ".local/share/lfm3/bin/llama-cli"


def _resolve_default_vcache_run() -> Path:
    found = shutil.which("vcache-run")
    if found:
        return Path(found)
    return Path(__file__).resolve().parent.parent.parent / "bin/vcache-run"


DEFAULT_LLAMA_CLI = _resolve_default_llama_cli()
DEFAULT_VCACHE_RUN = _resolve_default_vcache_run()

SCENARIOS: dict[str, dict] = {
    "refactor": {
        "name": "Code Refactoring (High N-Gram Overlap)",
        "prompt": (
            "pub struct AgentContext {\n"
            "    pub session_id: String,\n"
            "    pub conversation_id: String,\n"
            "    pub token_budget: usize,\n"
            "    pub tokens_used: usize,\n"
            "    pub tokens_remaining: usize,\n"
            "    pub is_active: bool,\n"
            "}\n\n"
            "// Re-emit the struct with identical fields:\n"
            "pub struct AgentContext {\n"
        ),
        "tokens": 65,
    },
    "boilerplate": {
        "name": "Repetitive Schema / DTO (Pattern Reuse)",
        "prompt": (
            "table UserMetrics {\n"
            "    metric_id: uint64;\n"
            "    user_id: uint64;\n"
            "    cache_hits: uint32;\n"
            "    cache_misses: uint32;\n"
            "    l3_residency_bytes: uint64;\n"
            "    dram_transfers_bytes: uint64;\n"
            "}\n\n"
            "// Generate matching SQL CREATE TABLE schema:\n"
            "CREATE TABLE user_metrics (\n"
        ),
        "tokens": 80,
    },
    "greenfield": {
        "name": "Greenfield Algorithm (Low Overlap)",
        "prompt": (
            "Implement an iterative binary search in Rust for sorted i32 slice:\n"
            f"fn binary_search(arr: &{chr(91)}i32{chr(93)}, target: i32) -> Option<usize> {{"
        ),
        "tokens": 70,
    },
}


@dataclass
class RunMetric:
    mode: str
    generation_speed: float
    prompt_speed: float
    acceptance_rate: float | None
    vram_before_mib: int
    vram_after_mib: int
    vram_delta_mib: int
    duration_s: float


@dataclass
class ScenarioResult:
    scenario_id: str
    scenario_name: str
    baseline: RunMetric
    unpinned: RunMetric
    l3_exploit: RunMetric
    speedup_unpinned: float
    speedup_l3: float
    prompt_speedup: float


def auto_detect_model() -> Path | None:
    """Dynamically discover suitable local GGUF model."""
    # 1. Check llama-swap config
    llama_swap_conf = Path.home() / ".config/llama-swap/config.yaml"
    if llama_swap_conf.exists():
        try:
            content = llama_swap_conf.read_text(encoding="utf-8")
            m = re.compile(r"-m\s+([^\s]+\.gguf|[^\s]+blobs/sha256-[0-9a-f]{64})").search(content)
            if m:
                candidate = Path(m.group(1))
                if candidate.exists():
                    return candidate
        except (OSError, UnicodeDecodeError):
            pass

    # 2. Check ollama blobs directly
    blob_dir = Path("/var/lib/ollama/blobs")
    if blob_dir.exists():
        for blob in blob_dir.glob("sha256-*"):
            if blob.is_file() and blob.stat().st_size > 2 * (1024**3):  # > 2GB
                return blob

    return None


def get_vram_used_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
        )
        return int(out.strip().split("\n")[0])
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0


class BenchmarkHarness:
    def __init__(
        self,
        model_path: Path,
        llama_cli: Path = DEFAULT_LLAMA_CLI,
        vcache_run: Path = DEFAULT_VCACHE_RUN,
    ):
        self.model_path = model_path
        self.llama_cli = llama_cli
        self.vcache_run = vcache_run
        self.topology = get_cpu_topology()

        if not self.llama_cli.exists():
            raise FileNotFoundError(f"llama-cli binary not found at {self.llama_cli}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF model not found at {self.model_path}")

    def _execute_run(self, prompt: str, tokens: int, mode: str) -> RunMetric:
        cmd = [
            str(self.llama_cli),
            "-m", str(self.model_path),
            "-ngl", "99",
            "-c", "4096",
            "-n", str(tokens),
            "--temp", "0",
            "-p", prompt,
            "-st",
            "--simple-io",
        ]

        prefix = []
        if mode == "baseline":
            cmd += ["--spec-type", "none"]
        elif mode == "unpinned_8c":
            cmd += [
                "--spec-type", "ngram-mod",
                "--spec-ngram-mod-n-min", "2",
                "--spec-ngram-mod-n-max", "16",
                "--spec-draft-threads", str(self.topology.physical_cores),
                "--spec-draft-ngl", "0",
            ]
        elif mode == "l3_exploit_4c":
            # Wrap with vcache-run if available
            if self.vcache_run.exists() and os.access(self.vcache_run, os.X_OK):
                prefix = [str(self.vcache_run), "--profile", "infer", "--"]
            cores = self.topology.recommended_infer_cores
            cmd += [
                "--spec-type", "ngram-mod",
                "--spec-ngram-mod-n-min", "2",
                "--spec-ngram-mod-n-max", "16",
                "--spec-draft-threads", "4",
                "--spec-draft-cpu-range", cores,
                "--spec-draft-cpu-strict", "1",
                "--spec-draft-ngl", "0",
            ]
        else:
            raise ValueError(f"Unknown benchmark mode: {mode}")

        vram_before = get_vram_used_mib()
        t0 = time.perf_counter()

        proc = subprocess.run(
            prefix + cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        duration = time.perf_counter() - t0
        vram_after = get_vram_used_mib()

        output = proc.stdout
        clean_output = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]").sub("", output)

        prompt_speed = 0.0
        gen_speed = 0.0
        acceptance_rate = None

        m_speed = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", clean_output)
        if m_speed:
            prompt_speed = float(m_speed.group(1))
            gen_speed = float(m_speed.group(2))

        # Check for draft acceptance rate telemetry
        m_accept = re.search(r"draft_acceptance\s*=\s*([\d\.]+)\s*%", clean_output)
        if m_accept:
            acceptance_rate = float(m_accept.group(1))

        return RunMetric(
            mode=mode,
            generation_speed=gen_speed,
            prompt_speed=prompt_speed,
            acceptance_rate=acceptance_rate,
            vram_before_mib=vram_before,
            vram_after_mib=vram_after,
            vram_delta_mib=max(0, vram_after - vram_before),
            duration_s=round(duration, 3),
        )

    def benchmark_scenario(self, scenario_id: str, runs: int = 1) -> ScenarioResult:
        sc = SCENARIOS[scenario_id]

        def get_best_metric(mode: str) -> RunMetric:
            metrics = [self._execute_run(sc["prompt"], sc["tokens"], mode) for _ in range(runs)]
            # Sort by generation speed descending
            metrics.sort(key=lambda m: m.generation_speed, reverse=True)
            return metrics[0]

        baseline = get_best_metric("baseline")
        unpinned = get_best_metric("unpinned_8c")
        l3_exploit = get_best_metric("l3_exploit_4c")

        base_gen = baseline.generation_speed or 1.0
        base_prompt = baseline.prompt_speed or 1.0

        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_name=sc["name"],
            baseline=baseline,
            unpinned=unpinned,
            l3_exploit=l3_exploit,
            speedup_unpinned=round(unpinned.generation_speed / base_gen, 2),
            speedup_l3=round(l3_exploit.generation_speed / base_gen, 2),
            prompt_speedup=round(l3_exploit.prompt_speed / base_prompt, 2),
        )


def format_table(results: list[ScenarioResult]) -> str:
    lines = [
        "=" * 104,
        f"{'WORKLOAD PROFILE':<26} | {'BASELINE (NO SPEC)':<18} | {'UNPINNED 8C SMT':<16} | {'L3 PINNED (4C)':<15} | {'SPEEDUP':<8} | {'VRAM Δ':<8}",
        "=" * 104,
    ]
    for r in results:
        base = f"{r.baseline.generation_speed:.1f} tok/s"
        unpin = f"{r.unpinned.generation_speed:.1f} tok/s"
        l3 = f"{r.l3_exploit.generation_speed:.1f} tok/s"
        speedup = f"{r.speedup_l3:.2f}x"
        vram = f"{r.l3_exploit.vram_delta_mib} MiB"
        lines.append(f"{r.scenario_id:<26} | {base:<18} | {unpin:<16} | {l3:<15} | {speedup:<8} | {vram:<8}")
    lines.append("=" * 104)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Zen 4 3D V-Cache Speculative Decoding Benchmark")
    parser.add_argument("--model", "-m", type=Path, default=None, help="GGUF model path (auto-detected if omitted)")
    parser.add_argument("--llama-cli", type=Path, default=DEFAULT_LLAMA_CLI, help="llama-cli binary path")
    parser.add_argument("--scenario", "-s", choices=list(SCENARIOS.keys()) + ["all"], default="all", help="Scenario")
    parser.add_argument("--runs", "-r", type=int, default=1, help="Number of benchmark iterations per mode")
    parser.add_argument("--format", "-f", choices=["table", "json", "csv"], default="table", help="Output format")
    args = parser.parse_args()

    model_path = args.model or auto_detect_model()
    if not model_path:
        sys.exit("Error: No GGUF model detected. Specify model explicitly with -m /path/to/model.gguf")

    harness = BenchmarkHarness(model_path=model_path, llama_cli=args.llama_cli)
    scenarios_to_run = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    if args.format == "table":
        model_display = model_path.name[:45]
        print("=" * 78)
        print(" ZEN 4 3D V-CACHE SPECULATIVE DRAFTING BENCHMARK")
        print(f" Model       : {model_display}...")
        print(f" CPU Topology: {harness.topology.model_name}")
        print(f" L3 3D Cache : {harness.topology.has_3d_vcache} (Pinning: Cores {harness.topology.recommended_infer_cores})")
        print("=" * 78)

    results: list[ScenarioResult] = []
    for sc_id in scenarios_to_run:
        if args.format == "table":
            sys.stdout.write(f"\n▶ Benchmarking scenario: {sc_id} ... ")
            sys.stdout.flush()
        res = harness.benchmark_scenario(sc_id, runs=args.runs)
        results.append(res)
        if args.format == "table":
            sys.stdout.write("Done.\n")

    if args.format == "table":
        print("\n" + format_table(results))
    elif args.format == "json":
        print(json.dumps([asdict(r) for r in results], indent=2))
    elif args.format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["scenario", "baseline_tps", "unpinned_tps", "l3_exploit_tps", "speedup", "vram_delta_mib"])
        for r in results:
            writer.writerow([
                r.scenario_id,
                r.baseline.generation_speed,
                r.unpinned.generation_speed,
                r.l3_exploit.generation_speed,
                r.speedup_l3,
                r.l3_exploit.vram_delta_mib,
            ])


if __name__ == "__main__":
    main()
