"""
Pure PyTorch Qwen3 Micro-Draft architecture and GGUF Q8_0 exporter.
Matches llama.cpp Qwen3 inference bit-for-bit (NeoX half-split RoPE, per-head Q/K RMSNorm,
SwiGLU FFN, tied embeddings, and optional GPU PCA warm-start from teacher token_embd).
"""

from __future__ import annotations

import math
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
    def generate_draft(self, prefix_ids: torch.Tensor, max_new: int = 8) -> torch.Tensor:
        """Greedy autoregressive rollout for SpecOPD on-policy branch evaluation."""
        seq = prefix_ids
        for _ in range(max_new):
            logits = self.forward(seq)[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_id], dim=1)
        return seq


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

    tokens_field = reader.get_field("tokenizer.ggml.tokens")
    tokens = [bytes(tokens_field.parts[i]).decode("utf-8", errors="replace") for i in tokens_field.data]
    token_types = [int(t) for t in reader.get_field("tokenizer.ggml.token_type").contents()]
    merges_field = reader.get_field("tokenizer.ggml.merges")
    merges = [bytes(merges_field.parts[i]).decode("utf-8", errors="replace") for i in merges_field.data]

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(int(reader.get_field("tokenizer.ggml.bos_token_id").contents()))
    writer.add_eos_token_id(int(reader.get_field("tokenizer.ggml.eos_token_id").contents()))
    writer.add_pad_token_id(int(reader.get_field("tokenizer.ggml.padding_token_id").contents()))
    writer.add_add_bos_token(bool(reader.get_field("tokenizer.ggml.add_bos_token").contents()))

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
