"""Unit tests for Qwen3MicroDraft architecture, presets, and SpecOPD training prompts."""

import unittest

from vcache.prompts_specopd import build_training_prompts
from vcache.specopd import L3_BUDGET_BYTES, SpecOpdMemoryBudget

try:
    import torch

    from vcache.qwen3_draft import Qwen3DraftConfig, Qwen3MicroDraft
    from vcache.train_specopd import _collate_batch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestPromptsSpecOpd(unittest.TestCase):
    def test_prompts_exclude_benchmark_prompts(self) -> None:
        for expanded, style, min_count in (
            (False, "qwen3", 45),
            (True, "qwen3", 140),
            (True, "deepseek", 140),
        ):
            prompts = build_training_prompts(expanded=expanded, style=style)
            self.assertGreaterEqual(len(prompts), min_count)
            self.assertEqual(len(prompts), len(set(prompts)))
            joined = "\n".join(prompts)
            for forbidden in ("struct AgentContext", "table UserMetrics", "fn binary_search"):
                self.assertNotIn(forbidden, joined)
        ds_prompts = build_training_prompts(expanded=True, style="deepseek")
        self.assertTrue(any("<｜User｜>" in p and "<｜Assistant｜>" in p for p in ds_prompts))


@unittest.skipUnless(HAS_TORCH, "torch not installed in system python")
class TestQwen3MicroDraft(unittest.TestCase):
    def test_presets_and_forward_shapes(self) -> None:
        for arch, embd in (("d64", 64), ("d128", 128), ("d192", 192)):
            cfg = Qwen3DraftConfig.preset(arch, vocab_size=256)
            self.assertEqual(cfg.n_embd, embd)
            model = Qwen3MicroDraft(cfg)
            x = torch.randint(0, 256, (2, 12))
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (2, 12, 256))
            rolled = model.generate_draft(x[:1, :4], max_new=5)
            self.assertEqual(tuple(rolled.shape), (1, 9))

    def test_d192_fits_l3_budget(self) -> None:
        cfg = Qwen3DraftConfig.preset("d192")
        model = Qwen3MicroDraft(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        budget = SpecOpdMemoryBudget(
            params_millions=(n_params / 1_000_000) * 1.0625,
            num_layers=cfg.n_layer,
            num_heads=cfg.n_head_kv,
            head_dim=cfg.head_dim,
            kv_bits=16,
            workspace_mib=2.0,
        )
        self.assertTrue(budget.fits_in_l3_budget(L3_BUDGET_BYTES))

    def test_collate_batch_alignment(self) -> None:
        rec = {
            "prompt": [1, 2, 3],
            "gen": [4, 5],
            "top_ids": [[4, 6], [5, 7]],
            "top_lps": [[-0.2, -2.0], [-0.1, -3.0]],
        }
        x, tgt, mask, top_ids, top_probs = _collate_batch([rec], top_k=2, device=torch.device("cpu"))
        self.assertEqual(x.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(tgt[mask].tolist(), [4, 5])
        self.assertEqual(top_ids.shape, (1, 4, 2))
        self.assertAlmostEqual(float(top_probs[mask][0].sum()), 1.0, places=5)

    def test_gguf_export_clones_teacher_tokenizer(self) -> None:
        import tempfile
        from pathlib import Path

        import gguf
        import numpy as np

        from vcache.qwen3_draft import export_to_gguf_q8

        vocab = 256
        tokens = [f"t{i}".encode() for i in range(vocab)]
        tokens[250] = "<｜User｜>".encode()
        tokens[251] = b"\xff\xfe"  # non-UTF-8 bytes must survive byte-exactly
        with tempfile.TemporaryDirectory() as tmp:
            teacher = Path(tmp) / "teacher.gguf"
            w = gguf.GGUFWriter(str(teacher), arch="qwen3")
            w.add_tokenizer_model("gpt2")
            w.add_tokenizer_pre("deepseek-r1-qwen")
            w.add_token_list(tokens)
            w.add_token_types([1] * vocab)
            w.add_token_merges([b"t 1", b"t 2"])
            w.add_bos_token_id(250)
            w.add_eos_token_id(3)
            w.add_add_bos_token(False)
            w.add_add_eos_token(False)  # no padding_token_id on purpose
            w.add_tensor("dummy", np.zeros(32, dtype=np.float32))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()

            out = Path(tmp) / "draft.gguf"
            export_to_gguf_q8(Qwen3MicroDraft(Qwen3DraftConfig.preset("d64", vocab_size=vocab)), teacher, out)
            f = gguf.GGUFReader(str(out), "r").fields
            tok = f["tokenizer.ggml.tokens"]
            self.assertEqual([bytes(tok.parts[i]) for i in tok.data], tokens)
            self.assertEqual(bytes(f["tokenizer.ggml.pre"].parts[-1]), b"deepseek-r1-qwen")
            self.assertEqual(int(f["tokenizer.ggml.bos_token_id"].contents()), 250)
            self.assertFalse(bool(f["tokenizer.ggml.add_eos_token"].contents()))
            self.assertNotIn("tokenizer.ggml.padding_token_id", f)

            with self.assertRaises(ValueError):
                export_to_gguf_q8(Qwen3MicroDraft(Qwen3DraftConfig.preset("d64", vocab_size=128)), teacher, out)

    def test_rollout_cache_key_separates_teachers(self) -> None:
        from pathlib import Path

        from vcache.train_specopd import TrainConfig, _rollout_cache_key

        qwen = TrainConfig(teacher_gguf=Path("qwen3-4b.gguf"), output_dir=Path("."), expanded_corpus=True)
        r1 = TrainConfig(teacher_gguf=Path("r1-8b.gguf"), output_dir=Path("."), expanded_corpus=True, prompt_style="deepseek")
        self.assertNotEqual(_rollout_cache_key(qwen), _rollout_cache_key(r1))


if __name__ == "__main__":
    unittest.main()
