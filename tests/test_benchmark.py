"""
Unit tests for Benchmark parsing and L3 Vector Engine.
"""

import unittest

from vcache.benchmark import RunMetric, ScenarioResult, format_table
from vcache.vector import L3VectorEngine


class TestBenchmark(unittest.TestCase):
    def test_format_table(self):
        m1 = RunMetric("baseline", 80.0, 2000.0, None, 1000, 1000, 0, 1.0)
        m2 = RunMetric("unpinned_8c", 130.0, 1950.0, None, 1000, 1000, 0, 0.7)
        m3 = RunMetric("l3_exploit_4c", 135.0, 2250.0, 65.5, 1000, 1000, 0, 0.6)
        res = ScenarioResult(
            scenario_id="refactor",
            scenario_name="Code Refactoring",
            baseline=m1,
            unpinned=m2,
            l3_exploit=m3,
            speedup_unpinned=1.63,
            speedup_l3=1.69,
            prompt_speedup=1.13,
        )
        table_str = format_table([res])
        self.assertIn("refactor", table_str)
        self.assertIn("1.69x", table_str)
        self.assertIn("80.0 tok/s", table_str)

    def test_vector_engine(self):
        engine = L3VectorEngine()
        vec_a = b"\x00" * 64
        vec_b = b"\xff" * 64
        # Hamming distance between all 0s and all 1s across 512 bits is 512
        dist = engine.batch_hamming(vec_a, vec_b, 1)
        self.assertEqual(dist[0], 512)

        # Distance with identical vector is 0
        dist_zero = engine.batch_hamming(vec_a, vec_a, 1)
        self.assertEqual(dist_zero[0], 0)


if __name__ == "__main__":
    unittest.main()
