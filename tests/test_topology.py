"""
Unit tests for CPU Topology and 3D V-Cache detection.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

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
        self.assertIn(topo.recommended_infer_cores, ["2-7", "4-7", "0-7"])

    def test_avx512_vbmi_flag_detection(self):
        fake_cpuinfo = (
            "model name\t: AMD Ryzen 7 7800X3D 8-Core Processor\n"
            "core id\t\t: 0\n"
            "flags\t\t: fpu avx512f avx512vbmi avx512_vbmi2 avx512_vnni\n"
        )
        orig_read_text = Path.read_text

        def fake_read_text(path_obj: Path, *args, **kwargs) -> str:
            if str(path_obj) == "/proc/cpuinfo":
                return fake_cpuinfo
            return orig_read_text(path_obj, *args, **kwargs)

        with patch.object(Path, "read_text", fake_read_text):
            topo = get_cpu_topology()
        self.assertTrue(topo.avx512_vbmi_supported)
        self.assertTrue(topo.avx512_vnni_supported)


if __name__ == "__main__":
    unittest.main()
