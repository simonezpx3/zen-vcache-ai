"""
Unit tests for CPU Topology and 3D V-Cache detection.
"""

import unittest

from vcache.topology import _parse_cpu_list, get_cpu_topology


class TestTopology(unittest.TestCase):
    def test_parse_cpu_list(self):
        self.assertEqual(_parse_cpu_list("0-3"), [0, 1, 2, 3])
        self.assertEqual(_parse_cpu_list("0-3,5,7-8"), [0, 1, 2, 3, 5, 7, 8])
        self.assertEqual(_parse_cpu_list(""), [])
        self.assertEqual(_parse_cpu_list("4"), [4])

    def test_get_cpu_topology(self):
        topo = get_cpu_topology()
        self.assertIsNotNone(topo.model_name)
        self.assertGreater(topo.total_logical_cpus, 0)
        self.assertGreater(topo.physical_cores, 0)
        self.assertIsInstance(topo.has_3d_vcache, bool)
        self.assertIn(topo.recommended_infer_cores, ["4-7", "0-7"])


if __name__ == "__main__":
    unittest.main()
