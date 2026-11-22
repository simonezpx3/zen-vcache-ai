#!/usr/bin/env python3
"""Evaluate Top-16K lm_head shortlist pruning on trained d192 draft vs DeepSeek-R1-8B teacher."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vcache.prompts_specopd import build_training_prompts
from vcache.qwen3_draft import build_shortlist_ids, load_from_gguf_q8
from vcache.train_specopd import LlamaTeacherClient

DEFAULT_TEACHER = Path.home() / ".local/share/models/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf"
DEFAULT_DRAFT = Path.home() / "models/qwen3-4b/qwen3-specopd-d192-r1_8b-q8_0.gguf"

HELD_OUT_TASKS = [
    ("refactor", "Refactor this Rust struct into an immutable summary with From<&Worker> impl:\npub struct Worker {\n    pub id: u32,\n    pub jobs_done: u64,\n    pub mean_us: f64,\n}\n"),
    ("refactor", "Rewrite this Python loop using a single comprehension and type hints:\ndef collect_valid(items):\n    out = []\n    for x in items:\n        if x is not None and x > 0:\n            out.append(int(x))\n    return out\n"),
    ("chat_rust", "Explain briefly how lock-free ring buffers avoid false sharing on x86_64 cache lines and show a minimal #[repr(align(64))] struct in Rust."),
    ("chat_rust", "Write a concise C function using AVX-512 _mm512_popcnt_epi64 to count differing bits across two 64-byte buffers."),
    ("greenfield", "Implement an LRU cache in Python 3.12 with O(1) get and put using collections.OrderedDict and unit assertions."),
    ("greenfield", "Write a binary search function in Rust returning Result<usize, usize> with full edge-case handling."),
]


def _format_deepseek_prompt(text: str) -> str:
    return f"<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜><think>\n</think>\n"


def _eval_draft_vs_teacher(
    model,
    rollouts: list[dict],
    shortlist: bool,
    rollout_steps: int = 4,
) -> dict:
    total_tokens = 0
    top1_hits = 0
    prefix_matches = 0
    prefix_total = 0
    draft_tokens_gen = 0
    t_ns = 0

    with torch.inference_mode():
        for rec in rollouts:
            seq = rec["prompt"] + rec["gen"]
            p_len = len(rec["prompt"])
            g_len = len(rec["gen"])
            if g_len == 0:
                continue

            full_t = torch.tensor([seq[:-1]], dtype=torch.long)
            t0 = time.perf_counter_ns()
            if shortlist:
                logits, sl_ids = model.forward_shortlist(full_t)
                local_pred = logits[0, p_len - 1 :, :].argmax(dim=-1)
                preds = sl_ids[local_pred].tolist()
            else:
                logits = model.forward(full_t)
                preds = logits[0, p_len - 1 :, :].argmax(dim=-1).tolist()
            t_ns += time.perf_counter_ns() - t0

            for j, tgt in enumerate(rec["gen"]):
                total_tokens += 1
                if preds[j] == tgt:
                    top1_hits += 1

            # Autoregressive rollout latency + prefix acceptance every rollout_steps
            for j in range(0, max(0, g_len - rollout_steps), rollout_steps):
                ctx = torch.tensor([seq[: p_len + j]], dtype=torch.long)
                t1 = time.perf_counter_ns()
                if shortlist:
                    out = model.generate_draft_shortlist(ctx, max_new=rollout_steps)
                else:
                    out = model.generate_draft(ctx, max_new=rollout_steps)
                t_ns += time.perf_counter_ns() - t1
                draft_tokens_gen += rollout_steps
                gen_slice = out[0, p_len + j :].tolist()
                tgt_slice = rec["gen"][j : j + rollout_steps]
                for a, b in zip(gen_slice, tgt_slice):
                    if a == b:
                        prefix_matches += 1
                    else:
                        break
                prefix_total += len(tgt_slice)

    sec = max(t_ns / 1e9, 1e-9)
    return {
        "top1_acc": round(top1_hits / max(1, total_tokens), 4),
        "prefix_accept": round(prefix_matches / max(1, prefix_total), 4),
        "eval_tokens": total_tokens,
        "draft_rollout_tokens": draft_tokens_gen,
        "cpu_tok_s": round((total_tokens + draft_tokens_gen) / sec, 1),
        "elapsed_s": round(sec, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Top-16K lm_head shortlist on R1-8B d192 draft")
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--shortlist-size", type=int, default=16_384)
    parser.add_argument("--calib-prompts", type=int, default=24)
    parser.add_argument("--max-predict", type=int, default=96)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    calib_prompts = build_training_prompts(expanded=True, style="deepseek")[: args.calib_prompts]
    eval_prompts = [_format_deepseek_prompt(p) for _, p in HELD_OUT_TASKS]

    calib_rollouts: list[dict] = []
    eval_rollouts: list[dict] = []
    with LlamaTeacherClient(args.teacher, ctx_size=1024) as client:
        for idx, p_text in enumerate(calib_prompts):
            ptoks = client.tokenize(p_text)
            rec = client.rollout_with_logprobs(ptoks, args.max_predict, temp=0.0, n_probs=4, seed=42 + idx)
            if rec is not None:
                calib_rollouts.append(rec)
        for idx, p_text in enumerate(eval_prompts):
            ptoks = client.tokenize(p_text)
            rec = client.rollout_with_logprobs(ptoks, args.max_predict, temp=0.0, n_probs=4, seed=1000 + idx)
            if rec is not None:
                eval_rollouts.append(rec)

    calib_seqs = [r["prompt"] + r["gen"] for r in calib_rollouts]
    sl_ids = build_shortlist_ids(
        teacher_gguf=args.teacher,
        token_sequences=calib_seqs,
        shortlist_size=args.shortlist_size,
    )
    sl_set = set(sl_ids.tolist())

    held_gen_tokens = [tok for r in eval_rollouts for tok in r["gen"]]
    covered = sum(1 for tok in held_gen_tokens if tok in sl_set)
    coverage = covered / max(1, len(held_gen_tokens))

    draft = load_from_gguf_q8(args.draft, device="cpu")
    draft.set_shortlist(sl_ids)

    # Warmup
    warm_t = torch.tensor([eval_rollouts[0]["prompt"][:16]], dtype=torch.long)
    with torch.inference_mode():
        _ = draft.forward(warm_t)
        _ = draft.forward_shortlist(warm_t)

    full_metrics = _eval_draft_vs_teacher(draft, eval_rollouts, shortlist=False)
    sl_metrics = _eval_draft_vs_teacher(draft, eval_rollouts, shortlist=True)

    result = {
        "teacher": args.teacher.name,
        "draft": args.draft.name,
        "threads": args.threads,
        "shortlist_size": int(sl_ids.numel()),
        "calib_rollouts": len(calib_rollouts),
        "calib_tokens": sum(len(r["gen"]) for r in calib_rollouts),
        "eval_rollouts": len(eval_rollouts),
        "eval_tokens": len(held_gen_tokens),
        "shortlist_coverage": round(coverage, 4),
        "lm_head_full_q8_mib": round(draft.lm_head_bytes(shortlist=False, q8=True) / (1024**2), 2),
        "lm_head_shortlist_q8_mib": round(draft.lm_head_bytes(shortlist=True, q8=True) / (1024**2), 2),
        "full_151k": full_metrics,
        "shortlist_16k": sl_metrics,
        "draft_speedup": round(sl_metrics["cpu_tok_s"] / max(0.1, full_metrics["cpu_tok_s"]), 2),
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
