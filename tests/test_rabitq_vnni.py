"""Unit and recall tests for Two-Tier L3-RaBitQ + INT8 VNNI search kernel."""

import ctypes
import unittest

import numpy as np

from vcache.vector import L3VectorEngine, TwoTierIVFRaBitQIndex, read_anon_hugepages_kb


class TestRaBitQVNNI(unittest.TestCase):
    def test_vnni_int8_dot_exact(self):
        engine = L3VectorEngine()
        rng = np.random.default_rng(123)
        for _ in range(20):
            q = rng.integers(-128, 128, size=512, dtype=np.int8)
            d = rng.integers(-127, 128, size=512, dtype=np.int8)
            doc_bias = int(-128 * d.astype(np.int32).sum())
            got = engine.int8_dot_vnni(q.ctypes.data, d.ctypes.data, doc_bias)
            expected = int(np.dot(q.astype(np.int32), d.astype(np.int32)))
            self.assertEqual(got, expected)

    def test_thp_alloc_and_smaps_reader(self):
        engine = L3VectorEngine()
        nbytes = 4 * 1024 * 1024
        ptr = engine.alloc_thp(nbytes)
        try:
            buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
            buf[0] = 7
            buf[nbytes - 1] = 9
            self.assertEqual(buf[0], 7)
            self.assertEqual(buf[nbytes - 1], 9)
            self.assertGreaterEqual(read_anon_hugepages_kb(), 0)
        finally:
            engine.free_thp(ptr)

    def test_two_tier_ivf_rabitq_recall_and_nta_parity(self):
        rng = np.random.default_rng(7)
        num_clusters = 128
        docs_per_cluster = 160
        dim = 512

        centers = rng.standard_normal((num_clusters, dim), dtype=np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

        docs = np.empty((num_clusters * docs_per_cluster, dim), dtype=np.float32)
        doc_sigma = np.float32(0.45 / np.sqrt(dim))
        query_sigma = np.float32(0.35 / np.sqrt(dim))
        for c in range(num_clusters):
            s = c * docs_per_cluster
            e = s + docs_per_cluster
            noise = doc_sigma * rng.standard_normal((docs_per_cluster, dim), dtype=np.float32)
            block = centers[c][None, :] + noise
            block /= np.linalg.norm(block, axis=1, keepdims=True)
            docs[s:e] = block

        index = TwoTierIVFRaBitQIndex(seed=42)
        try:
            meta = index.build(docs, num_centroids=num_clusters, niter=12)
            self.assertEqual(meta["num_docs"], docs.shape[0])
            self.assertGreater(meta["tier1_bytes"], 10 * 1024 * 1024)

            recalls = []
            for qi in range(25):
                cid = rng.integers(0, num_clusters)
                q = centers[cid] + query_sigma * rng.standard_normal(dim, dtype=np.float32)
                q /= np.linalg.norm(q)

                gt_scores = docs @ q
                gt_top10 = set(np.argsort(-gt_scores)[:10].tolist())

                ids_nta, scores_nta = index.search(
                    q, top_k=10, top_c_stage1=48, nprobe=12, use_nta=True
                )
                ids_t0, scores_t0 = index.search(
                    q, top_k=10, top_c_stage1=48, nprobe=12, use_nta=False
                )

                self.assertEqual(ids_nta.tolist(), ids_t0.tolist())
                self.assertEqual(scores_nta.tolist(), scores_t0.tolist())
                recalls.append(len(set(ids_nta.tolist()) & gt_top10) / 10.0)

            mean_recall = float(np.mean(recalls))
            self.assertGreaterEqual(mean_recall, 0.92, f"Recall@10 too low: {mean_recall:.3f}")
        finally:
            index.close()


if __name__ == "__main__":
    unittest.main()
