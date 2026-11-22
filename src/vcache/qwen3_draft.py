"""
Pure PyTorch Qwen3 Micro-Draft architecture and GGUF Q8_0 exporter.
Matches llama.cpp Qwen3 inference bit-for-bit (NeoX half-split RoPE, per-head Q/K RMSNorm,
SwiGLU FFN, tied embeddings, and optional GPU PCA warm-start from teacher token_embd).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Qwen3DraftConfig:
    """Hyperparameters for an L3-resident Qwen3 micro-draft model."""

    name: str = "Qwen3-SpecOPD-d64"
    vocab_size: int = 151_936
    n_layer: int = 4
    n_embd: int = 64
    n_ff: int = 256
    n_head: int = 4
    n_head_kv: int = 2
    head_dim: int = 32
    context_length: int = 4096
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6

    @classmethod
    def preset(cls, arch: str, vocab_size: int = 151_936) -> Qwen3DraftConfig:
        if arch == "d64":
            return cls(
                name="Qwen3-SpecOPD-d64",
                vocab_size=vocab_size,
                n_layer=4,
                n_embd=64,
                n_ff=256,
                n_head=4,
                n_head_kv=2,
                head_dim=32,
            )
        if arch == "d128":
            return cls(
                name="Qwen3-SpecOPD-d128",
                vocab_size=vocab_size,
                n_layer=4,
                n_embd=128,
                n_ff=512,
                n_head=4,
                n_head_kv=2,
                head_dim=64,
            )
        if arch == "d192":
            return cls(
                name="Qwen3-SpecOPD-d192",
                vocab_size=vocab_size,
                n_layer=4,
                n_embd=192,
                n_ff=768,
                n_head=4,
                n_head_kv=2,
                head_dim=64,
            )
        raise ValueError(f"Unsupported draft preset: {arch}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(orig)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_neox_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q: [B, H, T, D], k: [B, H_kv, T, D], cos/sin: [1, 1, T, D]
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3DraftConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_head_kv = cfg.n_head_kv
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_head_kv * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_head_kv * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * cfg.head_dim, cfg.n_embd, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_head, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.n_head_kv, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.n_head_kv, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)  # [B, H, T, D]
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = _apply_neox_rope(q, k, cos, sin)
        if self.n_head_kv != self.n_head:
            rep = self.n_head // self.n_head_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.n_head * self.head_dim)
        return self.o_proj(out)


class Qwen3Block(nn.Module):
    def __init__(self, cfg: Qwen3DraftConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.attn = Qwen3Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.ffn_gate = nn.Linear(cfg.n_embd, cfg.n_ff, bias=False)
        self.ffn_up = nn.Linear(cfg.n_embd, cfg.n_ff, bias=False)
        self.ffn_down = nn.Linear(cfg.n_ff, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        h = self.ffn_norm(x)
        x = x + self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return x


class Qwen3MicroDraft(nn.Module):
    def __init__(self, cfg: Qwen3DraftConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embd = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Qwen3Block(cfg) for _ in range(cfg.n_layer)])
        self.output_norm = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.register_buffer("shortlist_ids", None, persistent=False)
        self.register_buffer("shortlist_head_weight", None, persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.cfg.n_embd)
        nn.init.normal_(self.token_embd.weight, mean=0.0, std=0.02)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=std)

    def warm_start_embedding_pca(self, teacher_gguf: Path, device: torch.device) -> None:
        """Project teacher Q6_K/F32 token_embd via GPU PCA to preserve full-vocab geometry."""
        reader = gguf.GGUFReader(str(teacher_gguf), "r")
        embd_tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
        teacher_f32 = gguf.quants.dequantize(embd_tensor.data, embd_tensor.tensor_type)
        with torch.no_grad():
            mat = torch.from_numpy(np.ascontiguousarray(teacher_f32, dtype=np.float32)).to(device)
            mean = mat.mean(dim=0, keepdim=True)
            centered = mat - mean
            u, s, _ = torch.pca_lowrank(centered, q=self.cfg.n_embd, center=False, niter=4)
            proj = (u * s.unsqueeze(0)).to(self.token_embd.weight.dtype)
            # Rescale variance so initial logit spread matches student hidden scale (Layer 0 RMSNorm invariant)
            proj = proj * (0.16 / (proj.std().clamp(min=1e-6)))
            self.token_embd.weight.copy_(proj.to(self.token_embd.weight.device))

    def _rope_tables(self, seqlen: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.cfg.head_dim // 2
        inv_freq = 1.0 / (
            self.cfg.rope_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) * 2.0 / self.cfg.head_dim)
        )
        pos = torch.arange(seqlen, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(1, 1, seqlen, self.cfg.head_dim).to(dtype)
        sin = emb.sin().view(1, 1, seqlen, self.cfg.head_dim).to(dtype)
        return cos, sin

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embd(input_ids)
        cos, sin = self._rope_tables(input_ids.shape[1], x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.output_norm(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_hidden(input_ids)
        return F.linear(hidden, self.token_embd.weight)

    @torch.no_grad()
    def set_shortlist(self, shortlist_ids: torch.Tensor) -> None:
        """Install an active Top-K token shortlist for pruned lm_head evaluation."""
        ids = torch.unique(shortlist_ids.to(dtype=torch.long, device=self.token_embd.weight.device))
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("shortlist_ids must be a non-empty 1D tensor")
        if int(ids.min()) < 0 or int(ids.max()) >= self.cfg.vocab_size:
            raise ValueError("shortlist_ids out of vocabulary bounds")
        self.shortlist_ids = ids
        self.shortlist_head_weight = self.token_embd.weight.index_select(0, ids).detach().contiguous()

    def forward_shortlist(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shortlist_ids is None or self.shortlist_head_weight is None:
            raise RuntimeError("Shortlist not initialized; call set_shortlist() first")
        hidden = self.forward_hidden(input_ids)
        return F.linear(hidden, self.shortlist_head_weight), self.shortlist_ids

    def lm_head_bytes(self, shortlist: bool = False, q8: bool = True) -> int:
        rows = int(self.shortlist_ids.numel()) if (shortlist and self.shortlist_ids is not None) else self.cfg.vocab_size
        elems = rows * self.cfg.n_embd
        return (elems // 32) * 34 if q8 else elems * 4

    @torch.no_grad()
    def generate_draft(self, prefix_ids: torch.Tensor, max_new: int = 8) -> torch.Tensor:
        """Greedy autoregressive rollout for SpecOPD on-policy branch evaluation."""
        seq = prefix_ids
        for _ in range(max_new):
            logits = self.forward(seq)[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_id], dim=1)
        return seq

    @torch.no_grad()
    def generate_draft_shortlist(self, prefix_ids: torch.Tensor, max_new: int = 8) -> torch.Tensor:
        """Greedy autoregressive rollout using the pruned Top-K shortlist lm_head."""
        if self.shortlist_ids is None or self.shortlist_head_weight is None:
            raise RuntimeError("Shortlist not initialized; call set_shortlist() first")
        seq = prefix_ids
        for _ in range(max_new):
            logits, ids = self.forward_shortlist(seq)
            local_idx = logits[:, -1, :].argmax(dim=-1)
            next_id = ids[local_idx].unsqueeze(-1)
            seq = torch.cat([seq, next_id], dim=1)
        return seq


def _copy_teacher_tokenizer(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, vocab_size: int) -> None:
    """Clone the teacher tokenizer byte-exactly so llama.cpp accepts the draft for speculative decoding.

    llama.cpp's draft compatibility check compares vocab type, BOS/EOS ids, add_bos/add_eos flags and
    every token string, so model/pre and special-token metadata come from the teacher (DeepSeek-R1-Qwen3
    and Qwen3 differ in 4 special tokens) and tokens/merges are copied as raw bytes, never re-decoded.
    """

    def raw_strings(key: str) -> list[bytes]:
        field = reader.get_field(key)
        return [bytes(field.parts[i]) for i in field.data]

    tokens = raw_strings("tokenizer.ggml.tokens")
    if len(tokens) != vocab_size:
        raise ValueError(f"Teacher vocab has {len(tokens)} tokens but draft vocab_size={vocab_size}")

    writer.add_tokenizer_model(bytes(reader.get_field("tokenizer.ggml.model").parts[-1]).decode())
    pre_field = reader.get_field("tokenizer.ggml.pre")
    if pre_field is not None:
        writer.add_tokenizer_pre(bytes(pre_field.parts[-1]).decode())
    writer.add_token_list(tokens)
    writer.add_token_types([int(t) for t in reader.get_field("tokenizer.ggml.token_type").contents()])
    if reader.get_field("tokenizer.ggml.merges") is not None:
        writer.add_token_merges(raw_strings("tokenizer.ggml.merges"))

    for key in ("bos_token_id", "eos_token_id", "padding_token_id", "unknown_token_id", "eot_token_id"):
        field = reader.get_field(f"tokenizer.ggml.{key}")
        if field is not None:
            writer.add_uint32(f"tokenizer.ggml.{key}", int(field.contents()))
    for key in ("add_bos_token", "add_eos_token"):
        field = reader.get_field(f"tokenizer.ggml.{key}")
        if field is not None:
            writer.add_bool(f"tokenizer.ggml.{key}", bool(field.contents()))


def export_to_gguf_q8(model: Qwen3MicroDraft, teacher_gguf: Path, output_path: Path) -> Path:
    """Export trained PyTorch Qwen3MicroDraft to llama.cpp-compatible Q8_0 GGUF."""
    cfg = model.cfg
    reader = gguf.GGUFReader(str(teacher_gguf), "r")
    writer = gguf.GGUFWriter(str(output_path), arch="qwen3")

    writer.add_name(cfg.name)
    writer.add_uint32("qwen3.block_count", cfg.n_layer)
    writer.add_uint32("qwen3.context_length", cfg.context_length)
    writer.add_uint32("qwen3.embedding_length", cfg.n_embd)
    writer.add_uint32("qwen3.feed_forward_length", cfg.n_ff)
    writer.add_uint32("qwen3.attention.head_count", cfg.n_head)
    writer.add_uint32("qwen3.attention.head_count_kv", cfg.n_head_kv)
    writer.add_uint32("qwen3.attention.key_length", cfg.head_dim)
    writer.add_uint32("qwen3.attention.value_length", cfg.head_dim)
    writer.add_float32("qwen3.rope.freq_base", cfg.rope_theta)
    writer.add_float32("qwen3.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    writer.add_quantization_version(2)

    _copy_teacher_tokenizer(reader, writer, cfg.vocab_size)

    def add_f32(name: str, tensor: torch.Tensor) -> None:
        writer.add_tensor(name, tensor.detach().float().cpu().numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)

    def add_q8(name: str, tensor: torch.Tensor) -> None:
        arr = np.ascontiguousarray(tensor.detach().float().cpu().numpy(), dtype=np.float32)
        q8 = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
        writer.add_tensor(name, q8, raw_dtype=gguf.GGMLQuantizationType.Q8_0)

    add_q8("token_embd.weight", model.token_embd.weight)
    add_f32("output_norm.weight", model.output_norm.weight)

    for i, blk in enumerate(model.blocks):
        add_f32(f"blk.{i}.attn_norm.weight", blk.attn_norm.weight)
        add_f32(f"blk.{i}.ffn_norm.weight", blk.ffn_norm.weight)
        add_f32(f"blk.{i}.attn_q_norm.weight", blk.attn.q_norm.weight)
        add_f32(f"blk.{i}.attn_k_norm.weight", blk.attn.k_norm.weight)
        add_q8(f"blk.{i}.attn_q.weight", blk.attn.q_proj.weight)
        add_q8(f"blk.{i}.attn_k.weight", blk.attn.k_proj.weight)
        add_q8(f"blk.{i}.attn_v.weight", blk.attn.v_proj.weight)
        add_q8(f"blk.{i}.attn_output.weight", blk.attn.o_proj.weight)
        add_q8(f"blk.{i}.ffn_gate.weight", blk.ffn_gate.weight)
        add_q8(f"blk.{i}.ffn_up.weight", blk.ffn_up.weight)
        add_q8(f"blk.{i}.ffn_down.weight", blk.ffn_down.weight)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return output_path


def load_from_gguf_q8(gguf_path: Path, device: str | torch.device = "cpu") -> Qwen3MicroDraft:
    """Load a Qwen3MicroDraft model exported by export_to_gguf_q8() into PyTorch."""
    reader = gguf.GGUFReader(str(gguf_path), "r")

    def kv_int(key: str, default: int) -> int:
        f = reader.get_field(key)
        return int(f.contents()) if f is not None else default

    def kv_float(key: str, default: float) -> float:
        f = reader.get_field(key)
        return float(f.contents()) if f is not None else default

    tensors = {t.name: t for t in reader.tensors}
    embd_arr = gguf.quants.dequantize(tensors["token_embd.weight"].data, tensors["token_embd.weight"].tensor_type)
    vocab_size, n_embd = embd_arr.shape
    ffn_arr = gguf.quants.dequantize(tensors["blk.0.ffn_gate.weight"].data, tensors["blk.0.ffn_gate.weight"].tensor_type)
    n_ff = int(ffn_arr.shape[0])

    name_f = reader.get_field("general.name")
    name = bytes(name_f.parts[-1]).decode("utf-8", errors="replace") if name_f is not None else gguf_path.stem

    cfg = Qwen3DraftConfig(
        name=name,
        vocab_size=int(vocab_size),
        n_layer=kv_int("qwen3.block_count", 4),
        n_embd=int(n_embd),
        n_ff=n_ff,
        n_head=kv_int("qwen3.attention.head_count", 4),
        n_head_kv=kv_int("qwen3.attention.head_count_kv", 2),
        head_dim=kv_int("qwen3.attention.key_length", 64),
        context_length=kv_int("qwen3.context_length", 4096),
        rope_theta=kv_float("qwen3.rope.freq_base", 1_000_000.0),
        rms_norm_eps=kv_float("qwen3.attention.layer_norm_rms_epsilon", 1e-6),
    )
    model = Qwen3MicroDraft(cfg).to(device)

    def load_tensor(dst: torch.Tensor, tname: str) -> None:
        t = tensors[tname]
        arr = gguf.quants.dequantize(t.data, t.tensor_type)
        with torch.no_grad():
            dst.copy_(torch.from_numpy(np.array(arr, dtype=np.float32, copy=True)).to(device=dst.device, dtype=dst.dtype))

    with torch.no_grad():
        load_tensor(model.token_embd.weight, "token_embd.weight")
        load_tensor(model.output_norm.weight, "output_norm.weight")
        for i, blk in enumerate(model.blocks):
            load_tensor(blk.attn_norm.weight, f"blk.{i}.attn_norm.weight")
            load_tensor(blk.ffn_norm.weight, f"blk.{i}.ffn_norm.weight")
            load_tensor(blk.attn.q_norm.weight, f"blk.{i}.attn_q_norm.weight")
            load_tensor(blk.attn.k_norm.weight, f"blk.{i}.attn_k_norm.weight")
            load_tensor(blk.attn.q_proj.weight, f"blk.{i}.attn_q.weight")
            load_tensor(blk.attn.k_proj.weight, f"blk.{i}.attn_k.weight")
            load_tensor(blk.attn.v_proj.weight, f"blk.{i}.attn_v.weight")
            load_tensor(blk.attn.o_proj.weight, f"blk.{i}.attn_output.weight")
            load_tensor(blk.ffn_gate.weight, f"blk.{i}.ffn_gate.weight")
            load_tensor(blk.ffn_up.weight, f"blk.{i}.ffn_up.weight")
            load_tensor(blk.ffn_down.weight, f"blk.{i}.ffn_down.weight")
    model.eval()
    return model


def build_shortlist_ids(
    teacher_gguf: Path | None = None,
    token_sequences: list[list[int]] | None = None,
    shortlist_size: int = 16_384,
    vocab_size: int = 151_936,
) -> torch.Tensor:
    """Select Top-K active vocabulary IDs preserving special/byte tokens + corpus frequency."""
    pinned: list[int] = []
    seen: set[int] = set()

    def _add(tid: int) -> None:
        if 0 <= tid < vocab_size and tid not in seen:
            seen.add(tid)
            pinned.append(tid)

    if teacher_gguf is not None and teacher_gguf.is_file():
        reader = gguf.GGUFReader(str(teacher_gguf), "r")
        tok_field = reader.get_field("tokenizer.ggml.tokens")
        if tok_field is not None:
            vocab_size = len(tok_field.data)
            for key in ("bos_token_id", "eos_token_id", "padding_token_id", "unknown_token_id", "eot_token_id"):
                sf = reader.get_field(f"tokenizer.ggml.{key}")
                if sf is not None:
                    _add(int(sf.contents()))
            tt_field = reader.get_field("tokenizer.ggml.token_type")
            if tt_field is not None:
                ttypes = [int(t) for t in tt_field.contents()]
                for tid, ttype in enumerate(ttypes):
                    if ttype in (3, 4, 6):  # CONTROL / USER_DEFINED / BYTE
                        _add(tid)
            for tid in range(min(256, vocab_size)):
                _add(tid)

    counts: Counter[int] = Counter()
    if token_sequences:
        for seq in token_sequences:
            counts.update(seq)

    for tid, _freq in counts.most_common():
        if len(pinned) >= shortlist_size:
            break
        _add(int(tid))

    if len(pinned) < shortlist_size:
        for tid in range(vocab_size):
            if len(pinned) >= shortlist_size:
                break
            _add(tid)

    selected = sorted(pinned[:shortlist_size])
    return torch.tensor(selected, dtype=torch.long)
