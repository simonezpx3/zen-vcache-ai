"""
Unit tests for SpecOPD L3 3D V-Cache Draft Engine (arXiv:2611.02846).
Covers proposal tree masks, E2E-TV surrogate loss, 35 MiB L3 SRAM memory budget,
hardware perf watchdog, KV cache sanitization, and deterministic N-Gram fallback.
"""

import unittest

from vcache.specopd import (
    L3_BUDGET_BYTES,
    HybridSpecOpdDrafter,
    L3CacheWatchdog,
    NGramFallbackPredictor,
    ProposalTree,
    SpecOpdMemoryBudget,
    compute_e2e_tv_acceptance,
    evaluate_specopd_vs_ngram,
)


class TestSpecOpdL3Engine(unittest.TestCase):
    def test_proposal_tree_mask_topology(self):
        # Root (0) -> Branch A: 1 -> 2
        #          -> Branch B: 3 -> 4
        parents = [-1, 0, 1, 0, 3]
        tree = ProposalTree(parents=parents)
        self.assertEqual(tree.num_nodes, 5)

        masks = tree.build_bitmasks()
        # Node 0 attends only to 0 (0b00001 = 1)
        self.assertEqual(masks[0], 0b00001)
        # Node 1 attends to 0, 1 (0b00011 = 3)
        self.assertEqual(masks[1], 0b00011)
        # Node 2 attends to 0, 1, 2 (0b00111 = 7)
        self.assertEqual(masks[2], 0b00111)
        # Node 3 attends to 0, 3 (0b01001 = 9) - isolated from Branch A (1, 2)
        self.assertEqual(masks[3], 0b01001)
        # Node 4 attends to 0, 3, 4 (0b11001 = 25)
        self.assertEqual(masks[4], 0b11001)

    def test_avx512_tree_attention_int8(self):
        parents = [-1, 0, 0]
        tree = ProposalTree(parents=parents)
        # 3 nodes, 512-dim INT8 vectors
        q_vecs = [
            bytes([1] * 512),
            bytes([2] * 512),
            bytes([3] * 512),
        ]
        k_vecs = [
            bytes([1] * 512),
            bytes([1] * 512),
            bytes([1] * 512),
        ]
        scores = tree.compute_masked_scores_int8(q_vecs, k_vecs)
        # Node 0 -> Node 0 allowed (512 * 1 * 1 = 512), Node 0 -> 1, 2 masked
        self.assertEqual(scores[0][0], 512)
        self.assertLess(scores[0][1], -1_000_000_000)
        self.assertLess(scores[0][2], -1_000_000_000)

        # Node 1 -> Node 0, 1 allowed (512 * 2 * 1 = 1024), Node 1 -> 2 masked
        self.assertEqual(scores[1][0], 1024)
        self.assertEqual(scores[1][1], 1024)
        self.assertLess(scores[1][2], -1_000_000_000)

        # Node 2 -> Node 0, 2 allowed (512 * 3 * 1 = 1536), Node 2 -> 1 masked (sibling branch)
        self.assertEqual(scores[2][0], 1536)
        self.assertLess(scores[2][1], -1_000_000_000)
        self.assertEqual(scores[2][2], 1536)

    def test_e2e_tv_loss_and_acceptance_length(self):
        # Identical target and draft distributions -> TV = 0.0, acceptance alpha = 1.0
        p_target = [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.8, 0.1, 0.1]]
        q_draft_perfect = [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.8, 0.1, 0.1]]
        metrics_perfect = compute_e2e_tv_acceptance(p_target, q_draft_perfect)
        self.assertAlmostEqual(metrics_perfect["mean_tv_loss"], 0.0, places=6)
        self.assertAlmostEqual(metrics_perfect["mean_acceptance_rate"], 1.0, places=6)
        self.assertAlmostEqual(metrics_perfect["expected_acceptance_length"], 3.0, places=6)

        # Divergent draft distributions -> higher TV loss, lower expected acceptance length
        q_draft_divergent = [[0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]
        metrics_div = compute_e2e_tv_acceptance(p_target, q_draft_divergent)
        self.assertGreater(metrics_div["mean_tv_loss"], 0.4)
        self.assertLess(metrics_div["expected_acceptance_length"], 1.5)

    def test_l3_memory_budget_validation(self):
        # Valid micro-draft config: 12M params INT8, 4 layers, 8 heads, head_dim 64, 256 ring-buffer KV tokens
        budget_ok = SpecOpdMemoryBudget(
            params_millions=12.0,
            weight_bits=8,
            num_layers=4,
            num_heads=8,
            head_dim=64,
            max_kv_tokens=256,
            sink_tokens=4,
            kv_bits=8,
            workspace_mib=2.0,
        )
        self.assertTrue(budget_ok.fits_in_l3_budget())
        self.assertLessEqual(budget_ok.total_bytes, L3_BUDGET_BYTES)
        self.assertEqual(budget_ok.sliding_window_tokens, 252)

        # Oversized 128M model in INT8 (128 MiB weights > 35 MiB L3 budget) must be rejected
        budget_overflow = SpecOpdMemoryBudget(
            params_millions=128.0,
            weight_bits=8,
            num_layers=12,
            num_heads=12,
            head_dim=64,
            max_kv_tokens=2048,
        )
        self.assertFalse(budget_overflow.fits_in_l3_budget())
        self.assertGreater(budget_overflow.total_bytes, L3_BUDGET_BYTES)

    def test_kv_ring_buffer_and_watchdog_fallback(self):
        budget = SpecOpdMemoryBudget(params_millions=10.0, max_kv_tokens=256, sink_tokens=4)
        watchdog = L3CacheWatchdog(
            budget=budget,
            max_miss_rate=0.30,
            hard_token_limit=256,
        )
        ngram = NGramFallbackPredictor(n_min=2, n_max=5)
        drafter = HybridSpecOpdDrafter(budget=budget, watchdog=watchdog, fallback=ngram)

        context_tokens = [10, 20, 30, 40, 10, 20]
        # 1. Within budget & low L3 miss rate -> uses SpecOPD neural tree mode
        res_normal = drafter.draft_step(context_tokens, simulated_miss_rate=0.015)
        self.assertEqual(res_normal["engine"], "specopd_tree")
        self.assertFalse(res_normal["fallback_triggered"])

        # 2. High L3 miss rate (> 30% contention) -> triggers deterministic fallback to N-Gram
        res_contention = drafter.draft_step(context_tokens, simulated_miss_rate=0.42)
        self.assertEqual(res_contention["engine"], "ngram_fallback")
        self.assertTrue(res_contention["fallback_triggered"])
        self.assertEqual(res_contention["fallback_reason"], "l3_miss_rate_exceeded")
        # N-gram matches [10, 20] -> predicts [30, 40]
        self.assertEqual(res_contention["draft_tokens"][:2], [30, 40])

        # 3. Context exceeds hard KV token limit without ring-buffer wrap -> fallback
        watchdog.current_kv_tokens = 260
        res_kv_overflow = drafter.draft_step(context_tokens, simulated_miss_rate=0.01)
        self.assertEqual(res_kv_overflow["engine"], "ngram_fallback")
        self.assertEqual(res_kv_overflow["fallback_reason"], "kv_token_limit_exceeded")

        # 4. Sanitize KV buffer via AVX-512 + clflushopt zeroes all bytes
        kv_buf = bytearray(b"\xaa" * 1024)
        drafter.sanitize_kv_cache(kv_buf)
        self.assertEqual(kv_buf, bytearray(1024))

    def test_comparative_evaluation_vs_ngram(self):
        report = evaluate_specopd_vs_ngram()
        self.assertIn("scenarios", report)
        scenarios = {s["scenario_id"]: s for s in report["scenarios"]}
        self.assertIn("refactor", scenarios)
        self.assertIn("boilerplate", scenarios)
        self.assertIn("greenfield", scenarios)

        # On greenfield code, SpecOPD must substantially outperform N-Gram acceptance rate
        gf = scenarios["greenfield"]
        self.assertGreater(gf["specopd_acceptance_rate"], gf["ngram_acceptance_rate"] + 40.0)
        self.assertGreater(gf["specopd_speedup"], 1.5)
        self.assertLess(gf["l3_miss_rate_pct"], 2.0)
        self.assertEqual(gf["vram_overhead_mib"], 0)


if __name__ == "__main__":
    unittest.main()
