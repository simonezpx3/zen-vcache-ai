"""
Zen 4 3D V-Cache AI Exploitation Suite
Hardware-informed speculative drafting, 4-core saturation laws, and L3 vector acceleration.
"""

from .benchmark import BenchmarkHarness, RunMetric, ScenarioResult
from .specopd import (
    L3_BUDGET_BYTES,
    HybridSpecOpdDrafter,
    L3CacheWatchdog,
    NGramFallbackPredictor,
    ProposalTree,
    SpecOpdMemoryBudget,
    compute_e2e_tv_acceptance,
    evaluate_specopd_vs_ngram,
)
from .topology import CpuTopology, get_cpu_topology
from .vector import L3VectorEngine

__version__ = "1.2.0"
__all__ = [
    "L3_BUDGET_BYTES",
    "BenchmarkHarness",
    "CpuTopology",
    "HybridSpecOpdDrafter",
    "L3CacheWatchdog",
    "L3VectorEngine",
    "NGramFallbackPredictor",
    "ProposalTree",
    "RunMetric",
    "ScenarioResult",
    "SpecOpdMemoryBudget",
    "compute_e2e_tv_acceptance",
    "evaluate_specopd_vs_ngram",
    "get_cpu_topology",
]
