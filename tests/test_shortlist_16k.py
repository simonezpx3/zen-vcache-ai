"""Unit tests for Top-16K shortlist lm_head pruning and GGUF Q8_0 loading."""

import tempfile
import unittest
from pathlib import Path

try:
    import gguf
    import numpy as np
    import torch

    from vcache.qwen3_draft import (
        Qwen3DraftConfig,
        Qwen3MicroDraft,
        build_shortlist_ids,
        export_to_gguf_q8,
        load_from_gguf_q8,
    )

    HAS_TORCH_GGUF = True
except ImportError:
    HAS_TORCH_GGUF = False


@unittest.skipUnless(HAS_TORCH_GGUF, "torch/gguf not installed")
class TestShortlist16K(unittest.TestCase):
    def test_gguf_roundtrip_and_shortlist_parity(self) -> None:
        vocab_size = 512
        cfg = Qwen3DraftConfig.preset("d64", vocab_size=vocab_size)
        model = Qwen3MicroDraft(cfg)
        model.eval()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            teacher_path = tmp_dir / "teacher.gguf"
            draft_path = tmp_dir / "draft.gguf"

            tw = gguf.GGUFWriter(str(teacher_path), arch="qwen3")
            tw.add_tokenizer_model("gpt2")
            tw.add_tokenizer_pre("qwen2")
            tw.add_token_list([f"tok_{i}".encode() for i in range(vocab_size)])
            tw.add_token_types([1] * vocab_size)
            tw.add_uint32("tokenizer.ggml.bos_token_id", 1)
            tw.add_uint32("tokenizer.ggml.eos_token_id", 2)
            tw.add_tensor("dummy", np.zeros((4, 4), dtype=np.float32))
            tw.write_header_to_file()
            tw.write_kv_data_to_file()
            tw.write_tensors_to_file()
            tw.close()

            export_to_gguf_q8(model, teacher_path, draft_path)
            loaded = load_from_gguf_q8(draft_path, device="cpu")
            self.assertEqual(loaded.cfg.vocab_size, vocab_size)
            self.assertEqual(loaded.cfg.n_embd, 64)

            seq = [[1, 2, 10, 20, 30, 40]]
            sl_ids = build_shortlist_ids(teacher_path, token_sequences=seq, shortlist_size=128)
            self.assertEqual(int(sl_ids.numel()), 128)
            self.assertIn(1, sl_ids.tolist())
            self.assertIn(2, sl_ids.tolist())

            loaded.set_shortlist(sl_ids)
            self.assertLess(loaded.lm_head_bytes(shortlist=True), loaded.lm_head_bytes(shortlist=False))

            # Force winning logit inside shortlist by checking parity whenever argmax is in shortlist
            x = torch.tensor([[1, 10, 20]], dtype=torch.long)
            with torch.inference_mode():
                full_logits = loaded.forward(x)
                sl_logits, ids = loaded.forward_shortlist(x)
                self.assertEqual(tuple(sl_logits.shape), (1, 3, 128))
                # Every shortlist logit must match full_logits indexed at ids
                expected = full_logits.index_select(-1, ids)
                self.assertTrue(torch.allclose(sl_logits, expected, atol=1e-4))

                rolled = loaded.generate_draft_shortlist(x, max_new=4)
                self.assertEqual(tuple(rolled.shape), (1, 7))
                for tok in rolled[0, 3:].tolist():
                    self.assertIn(tok, set(ids.tolist()))

    def test_real_r1_8b_d192_shortlist_footprint(self) -> None:
        draft_path = Path.home() / "models/qwen3-4b/qwen3-specopd-d192-r1_8b-q8_0.gguf"
        if not draft_path.is_file():
            self.skipTest("Trained d192 R1-8B draft GGUF not present")
        loaded = load_from_gguf_q8(draft_path, device="cpu")
        self.assertEqual(loaded.cfg.vocab_size, 151_936)
        self.assertEqual(loaded.cfg.n_embd, 192)
        sl_ids = torch.arange(16_384, dtype=torch.long)
        loaded.set_shortlist(sl_ids)
        full_mib = loaded.lm_head_bytes(shortlist=False, q8=True) / (1024**2)
        sl_mib = loaded.lm_head_bytes(shortlist=True, q8=True) / (1024**2)
        self.assertGreater(full_mib, 28.0)
        self.assertLess(sl_mib, 3.5)


if __name__ == "__main__":
    unittest.main()
