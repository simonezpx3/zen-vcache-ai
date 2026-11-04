"""
Zen 4 3D V-Cache AI Exploitation Suite
Hardware-informed speculative drafting, 4-core saturation laws, and L3 vector acceleration.
"""

from .benchmark import BenchmarkHarness, RunMetric, ScenarioResult
from .topology import CpuTopology, get_cpu_topology
from .vector import L3VectorEngine

__version__ = "1.1.0"
__all__ = [
    "BenchmarkHarness",
    "CpuTopology",
    "L3VectorEngine",
    "RunMetric",
    "ScenarioResult",
    "get_cpu_topology",
]
