"""
On-Policy SpecOPD Micro-Draft Trainer (Stage 1 Forward Top-K KD + Stage 2 On-Policy Reverse-KL).
Targets quantized llama.cpp GGUF teachers directly via /tokenize and /completion n_probs.
"""

from __future__ import annotations

import gc
import json
import os
import random
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F

from .prompts_specopd import build_training_prompts
from .qwen3_draft import Qwen3DraftConfig, Qwen3MicroDraft, export_to_gguf_q8


@dataclass
class TrainConfig:
    teacher_gguf: Path
    output_dir: Path
    port: int = 0
    top_k: int = 16
    max_predict: int = 144
    stage1_epochs: int = 45
    stage2_rounds: int = 3
    stage2_branches: int = 64
    branch_len: int = 6
    batch_size: int = 8
    lr: float = 2e-3
    weight_decay: float = 0.01
    seed: int = 42
    expanded_corpus: bool = False
    prompt_style: str = "qwen3"
    output_tag: str | None = None


class LlamaTeacherClient:
    """Lifecycle wrapper around llama-server for logprob distillation & on-policy probing."""

    def __init__(self, teacher_gguf: Path, port: int = 0, ctx_size: int = 1024) -> None:
        self.teacher_gguf = teacher_gguf
        if port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                port = int(s.getsockname()[1])
        self.base_url = f"http://127.0.0.1:{port}"
        self.port = port
        self.ctx_size = ctx_size
        self._proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> Self:
        cmd = [
            os.environ.get("VCACHE_LLAMA_SERVER", str(Path.home() / ".local/share/lfm3/bin/llama-server")),
            "-m", str(self.teacher_gguf),
            "-ngl", "99",
            "-c", str(self.ctx_size),
            "--port", str(self.port),
            "-t", "6",
        ]
        self._log_file = (Path.home() / "Ai-temp/llama_teacher_srv.log").open("w+", encoding="utf-8")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self._log_file, text=True)
        for _ in range(80):
            if self._proc.poll() is not None:
                self._log_file.seek(0)
                tail = self._log_file.read()[-400:]
                raise RuntimeError(f"llama-server exited ({self._proc.returncode}) on port {self.port}: {tail}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=1) as r:
                    if r.status == 200:
                        return self
            except (OSError, urllib.error.URLError):
                time.sleep(0.25)
        self._log_file.seek(0)
        tail = self._log_file.read()[-400:]
        raise RuntimeError(f"llama-server failed to become healthy on port {self.port}: {tail}")

    def __exit__(self, *_: object) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if hasattr(self, "_log_file") and self._log_file:
            self._log_file.close()

    def _post(self, endpoint: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())

    def tokenize(self, text: str) -> list[int]:
        return [int(x) for x in self._post("/tokenize", {"content": text})["tokens"]]

    def rollout_with_logprobs(
        self, prompt_tokens: list[int], n_predict: int, temp: float, n_probs: int, seed: int
    ) -> dict | None:
        res = self._post(
            "/completion",
            {
                "prompt": prompt_tokens,
                "n_predict": n_predict,
                "temperature": temp,
                "top_p": 0.95,
                "n_probs": n_probs,
                "return_tokens": True,
                "seed": seed,
            },
        )
        gen_tokens = [int(t) for t in res.get("tokens", [])]
        cprobs = res.get("completion_probabilities", [])
        if len(gen_tokens) < 4 or len(cprobs) != len(gen_tokens):
            return None
        step_ids: list[list[int]] = []
        step_lps: list[list[float]] = []
        for entry in cprobs:
            tops = entry.get("top_logprobs", [])
            ids = [int(x["id"]) for x in tops[:n_probs]]
            lps = [float(x["logprob"]) for x in tops[:n_probs]]
            if not ids:
                return None
            step_ids.append(ids)
            step_lps.append(lps)
        return {"prompt": prompt_tokens, "gen": gen_tokens, "top_ids": step_ids, "top_lps": step_lps}

    def query_next_topk(self, prefix_tokens: list[int], n_probs: int = 16) -> tuple[list[int], list[float]] | None:
        res = self._post(
            "/completion",
            {
                "prompt": prefix_tokens,
                "n_predict": 1,
                "temperature": 0.0,
                "n_probs": n_probs,
                "ignore_eos": True,
                "cache_prompt": True,
            },
        )
        cprobs = res.get("completion_probabilities") or []
        if not cprobs:
            return None
        tops = (cprobs[0].get("top_logprobs") or [])[:n_probs]
        if not tops:
            return None
        return [int(x["id"]) for x in tops], [float(x["logprob"]) for x in tops]


def _rollout_cache_key(cfg: TrainConfig) -> dict:
    """Identity of a rollout corpus; a cache recorded for another teacher/template must never be reused."""
    return {
        "teacher": cfg.teacher_gguf.name,
        "prompt_style": cfg.prompt_style,
        "expanded_corpus": cfg.expanded_corpus,
        "max_predict": cfg.max_predict,
        "top_k": cfg.top_k,
        "seed": cfg.seed,
    }


def collect_teacher_rollouts(cfg: TrainConfig, cache_path: Path) -> list[dict]:
    """Collect and cache teacher trajectories + top-K logprobs via llama-server."""
    key = _rollout_cache_key(cfg)
    meta_path = cache_path.with_name(cache_path.name + ".meta.json")
    if cache_path.exists():
        cached_key = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if cached_key == key:
            records = [json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
            if records:
                print(f"SpecOPD // Loaded {len(records)} cached rollouts from {cache_path}")
                return records
        else:
            print(f"SpecOPD // Ignoring stale rollout cache {cache_path} (key {cached_key} != {key})")

    prompts = build_training_prompts(expanded=cfg.expanded_corpus, style=cfg.prompt_style)
    temp_schedule = [(0.0, 1), (0.7, 101), (0.8, 202)]
    records: list[dict] = []
    t0 = time.perf_counter()
    with LlamaTeacherClient(cfg.teacher_gguf, port=cfg.port) as client:
        for idx, p_text in enumerate(prompts):
            p_toks = client.tokenize(p_text)
            for run_i, (temp, seed_off) in enumerate(temp_schedule):
                rec = client.rollout_with_logprobs(
                    p_toks, cfg.max_predict, temp=temp, n_probs=cfg.top_k, seed=cfg.seed + idx * 10 + seed_off
                )
                if rec is not None:
                    records.append(rec)
            if (idx + 1) % 10 == 0:
                tot_toks = sum(len(r["gen"]) for r in records)
                print(f"SpecOPD // Collected {idx + 1}/{len(prompts)} prompts -> {len(records)} rollouts ({tot_toks} tokens) in {time.perf_counter() - t0:.1f}s")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    meta_path.write_text(json.dumps(key))
    print(f"SpecOPD // Corpus: {len(records)} rollouts, {sum(len(r['gen']) for r in records)} teacher tokens")
    return records


def _collate_batch(batch: list[dict], top_k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(r["prompt"]) + len(r["gen"]) for r in batch)
    bsz = len(batch)
    tokens = torch.zeros((bsz, max_len), dtype=torch.long)
    loss_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    top_ids = torch.zeros((bsz, max_len, top_k), dtype=torch.long)
    top_probs = torch.zeros((bsz, max_len, top_k), dtype=torch.float32)
    targets = torch.zeros((bsz, max_len), dtype=torch.long)

    for b, rec in enumerate(batch):
        p = rec["prompt"]
        g = rec["gen"]
        seq = p + g
        tokens[b, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        p_len = len(p)
        for j, tok in enumerate(g):
            pos = p_len - 1 + j
            loss_mask[b, pos] = True
            targets[b, pos] = tok
            ids = rec["top_ids"][j]
            lps = rec["top_lps"][j]
            k = min(top_k, len(ids))
            top_ids[b, pos, :k] = torch.tensor(ids[:k], dtype=torch.long)
            probs = torch.softmax(torch.tensor(lps[:k], dtype=torch.float32), dim=-1)
            top_probs[b, pos, :k] = probs

    return (
        tokens[:, :-1].to(device),
        targets[:, :-1].to(device),
        loss_mask[:, :-1].to(device),
        top_ids[:, :-1].to(device),
        top_probs[:, :-1].to(device),
    )


def train_stage1_forward_kd(
    model: Qwen3MicroDraft,
    records: list[dict],
    cfg: TrainConfig,
    device: torch.device,
) -> None:
    """Warm-start micro-draft on teacher rollouts via CE + sparse top-K forward KL."""
    model.train()
    embd_params = [model.token_embd.weight]
    other_params = [p for n, p in model.named_parameters() if n != "token_embd.weight"]
    opt = torch.optim.AdamW(
        [{"params": other_params, "lr": cfg.lr}, {"params": embd_params, "lr": cfg.lr * 0.5}],
        weight_decay=cfg.weight_decay,
    )
    rng = random.Random(cfg.seed)
    steps_per_epoch = max(1, (len(records) + cfg.batch_size - 1) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.stage1_epochs * steps_per_epoch, eta_min=cfg.lr * 0.1)

    for epoch in range(cfg.stage1_epochs):
        shuffled = list(records)
        rng.shuffle(shuffled)
        total_loss = 0.0
        total_top1 = 0
        total_tokens = 0
        for i in range(0, len(shuffled), cfg.batch_size):
            x, tgt, mask, top_ids, top_probs = _collate_batch(shuffled[i : i + cfg.batch_size], cfg.top_k, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)  # [B, T, V]
                valid_logits = logits[mask]
                valid_tgt = tgt[mask]
                valid_ids = top_ids[mask]
                valid_p_t = top_probs[mask]

                log_p_s = F.log_softmax(valid_logits.float(), dim=-1)
                ce = F.nll_loss(log_p_s, valid_tgt)
                s_at_k = log_p_s.gather(-1, valid_ids)
                kl = -(valid_p_t * s_at_k).sum(dim=-1).mean() + (valid_p_t * torch.log(valid_p_t.clamp(min=1e-8))).sum(dim=-1).mean()
                loss = 0.5 * ce + 0.5 * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            with torch.no_grad():
                n = int(mask.sum().item())
                total_loss += float(loss.item()) * n
                total_top1 += int((valid_logits.argmax(dim=-1) == valid_ids[:, 0]).sum().item())
                total_tokens += n

        print(f"  [Stage 1 | {model.cfg.name}] epoch {epoch + 1:02d}/{cfg.stage1_epochs} loss={total_loss / total_tokens:.4f} top1_acc={100.0 * total_top1 / total_tokens:.2f}%")

    del opt, sched
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_stage2_on_policy(
    model: Qwen3MicroDraft,
    records: list[dict],
    cfg: TrainConfig,
    device: torch.device,
) -> None:
    """On-Policy SpecOPD refinement: sample student branches, query teacher probs, optimize Reverse-KL + top1 match."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rng = random.Random(cfg.seed + 999)
    embd_params = [model.token_embd.weight]
    other_params = [p for n, p in model.named_parameters() if n != "token_embd.weight"]
    opt = torch.optim.AdamW(
        [{"params": other_params, "lr": cfg.lr * 0.05}, {"params": embd_params, "lr": cfg.lr * 0.01}],
        weight_decay=0.01,
    )
    best_accept = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    with LlamaTeacherClient(cfg.teacher_gguf, port=cfg.port) as client:
        for rnd in range(cfg.stage2_rounds):
            model.eval()
            branch_samples: list[tuple[list[int], list[list[int]], list[list[float]], list[int]]] = []
            accept_hits = 0
            accept_total = 0

            chosen = rng.sample(records, k=min(cfg.stage2_branches, len(records)))
            for rec in chosen:
                full = rec["prompt"] + rec["gen"]
                if len(full) < len(rec["prompt"]) + 4:
                    continue
                cut = rng.randint(len(rec["prompt"]), max(len(rec["prompt"]), len(full) - cfg.branch_len - 2))
                prefix = full[:cut]
                inp = torch.tensor([prefix], dtype=torch.long, device=device)
                rolled = model.generate_draft(inp, max_new=cfg.branch_len)[0].tolist()
                draft_toks = rolled[cut:]

                t_ids_seq: list[list[int]] = []
                t_lps_seq: list[list[float]] = []
                tgts: list[int] = []
                still_matching = True
                valid_branch = True
                for step_i in range(cfg.branch_len):
                    sub_prefix = rolled[: cut + step_i]
                    queried = client.query_next_topk(sub_prefix, n_probs=cfg.top_k)
                    if queried is None:
                        valid_branch = False
                        break
                    t_ids, t_lps = queried
                    t_ids_seq.append(t_ids)
                    t_lps_seq.append(t_lps)
                    tgts.append(t_ids[0])
                    if still_matching and draft_toks[step_i] == t_ids[0]:
                        accept_hits += 1
                    else:
                        still_matching = False
                    accept_total += 1
                if valid_branch and len(tgts) == cfg.branch_len:
                    branch_samples.append((rolled, t_ids_seq, t_lps_seq, tgts))

            cur_accept = accept_hits / max(1, accept_total)
            mean_len = cfg.branch_len * cur_accept
            if cur_accept >= best_accept:
                best_accept = cur_accept
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            model.train()
            rkl_sum = 0.0
            n_steps = 0
            accum = 16
            opt.zero_grad(set_to_none=True)
            t_lookup = torch.empty((model.cfg.vocab_size,), device=device, dtype=torch.float32)

            for b_idx, (seq, t_ids_seq, t_lps_seq, tgts) in enumerate(branch_samples):
                x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
                start_pos = len(seq) - cfg.branch_len - 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)[0, start_pos : start_pos + cfg.branch_len, :].float()
                log_p_s = F.log_softmax(logits, dim=-1)
                p_s = log_p_s.exp()

                loss_terms = []
                for step_i in range(cfg.branch_len):
                    t_ids = torch.tensor(t_ids_seq[step_i], dtype=torch.long, device=device)
                    t_lps = torch.tensor(t_lps_seq[step_i], dtype=torch.float32, device=device)
                    s_top_ids = torch.topk(log_p_s[step_i], k=cfg.top_k).indices
                    union_ids = torch.unique(torch.cat([t_ids, s_top_ids]))

                    t_lookup.fill_(float(t_lps.min().item()) - 4.0)
                    t_lookup[t_ids] = t_lps

                    u_log_ps = log_p_s[step_i, union_ids]
                    u_ps = p_s[step_i, union_ids]
                    u_log_pt = t_lookup[union_ids]
                    rkl = (u_ps * (u_log_ps - u_log_pt)).sum()
                    ce_top1 = -log_p_s[step_i, tgts[step_i]]
                    loss_terms.append(0.5 * rkl + 0.5 * ce_top1)

                loss = torch.stack(loss_terms).mean()
                (loss / accum).backward()
                rkl_sum += float(loss.item())
                n_steps += 1

                if (b_idx + 1) % accum == 0 or (b_idx + 1) == len(branch_samples):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            print(
                f"  [Stage 2 On-Policy | {model.cfg.name}] round {rnd + 1}/{cfg.stage2_rounds} "
                f"loss={rkl_sum / max(1, n_steps):.4f} prefix_accept={100.0 * cur_accept:.2f}% "
                f"(avg_match={mean_len:.2f}/{cfg.branch_len})"
            )

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  [Stage 2 | {model.cfg.name}] restored best checkpoint (prefix_accept={100.0 * best_accept:.2f}%)")


def run_training(arches: list[str], cfg: TrainConfig, cache_path: Path) -> list[Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = collect_teacher_rollouts(cfg, cache_path)
    outputs: list[Path] = []
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for arch in arches:
        torch.manual_seed(cfg.seed)
        dcfg = Qwen3DraftConfig.preset(arch)
        model = Qwen3MicroDraft(dcfg).to(device)
        print(f"\nSpecOPD // Training {dcfg.name} ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params)")
        model.warm_start_embedding_pca(cfg.teacher_gguf, device)
        train_stage1_forward_kd(model, records, cfg, device)
        train_stage2_on_policy(model, records, cfg, device)
        tag = cfg.output_tag or "trained"
        out_gguf = cfg.output_dir / f"qwen3-specopd-{arch}-{tag}-q8_0.gguf"
        export_to_gguf_q8(model, cfg.teacher_gguf, out_gguf)
        size_mib = out_gguf.stat().st_size / (1024 * 1024)
        print(f"SpecOPD // Exported {out_gguf} ({size_mib:.2f} MiB)")
        outputs.append(out_gguf)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs
