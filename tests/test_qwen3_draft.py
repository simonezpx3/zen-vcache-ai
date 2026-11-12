"""Unit tests for Qwen3MicroDraft architecture, presets, and SpecOPD training prompts."""

import unittest

from vcache.prompts_specopd import build_training_prompts

try:
    import torch

    from vcache.qwen3_draft import Qwen3DraftConfig, Qwen3MicroDraft
    from vcache.train_specopd import _collate_batch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestPromptsSpecOpd(unittest.TestCase):
    def test_prompts_exclude_benchmark_prompts(self) -> None:
        prompts = build_training_prompts()
        self.assertGreaterEqual(len(prompts), 45)
        joined = "\n".join(prompts)
        for forbidden in ("struct AgentContext", "table UserMetrics", "fn binary_search"):
            self.assertNotIn(forbidden, joined)


@unittest.skipUnless(HAS_TORCH, "torch not installed in system python")
class TestQwen3MicroDraft(unittest.TestCase):
    def test_presets_and_forward_shapes(self) -> None:
        for arch, embd in (("d64", 64), ("d128", 128)):
            cfg = Qwen3DraftConfig.preset(arch, vocab_size=256)
            self.assertEqual(cfg.n_embd, embd)
            model = Qwen3MicroDraft(cfg)
            x = torch.randint(0, 256, (2, 12))
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (2, 12, 256))
            rolled = model.generate_draft(x[:1, :4], max_new=5)
            self.assertEqual(tuple(rolled.shape), (1, 9))

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


if __name__ == "__main__":
    unittest.main()
