"""
Zen 4/5 3D V-Cache AI Suite
Hardware-informed L3 memory residency, PMU watchdog, and AVX-512 vector acceleration.
"""

from importlib import import_module

__version__ = "2.1.0"
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

_EXPORT_MAP = {
    "BenchmarkHarness": (".benchmark", "BenchmarkHarness"),
    "RunMetric": (".benchmark", "RunMetric"),
    "ScenarioResult": (".benchmark", "ScenarioResult"),
    "L3_BUDGET_BYTES": (".specopd", "L3_BUDGET_BYTES"),
    "HybridSpecOpdDrafter": (".specopd", "HybridSpecOpdDrafter"),
    "L3CacheWatchdog": (".specopd", "L3CacheWatchdog"),
    "NGramFallbackPredictor": (".specopd", "NGramFallbackPredictor"),
    "ProposalTree": (".specopd", "ProposalTree"),
    "SpecOpdMemoryBudget": (".specopd", "SpecOpdMemoryBudget"),
    "compute_e2e_tv_acceptance": (".specopd", "compute_e2e_tv_acceptance"),
    "evaluate_specopd_vs_ngram": (".specopd", "evaluate_specopd_vs_ngram"),
    "CpuTopology": (".topology", "CpuTopology"),
    "get_cpu_topology": (".topology", "get_cpu_topology"),
    "L3VectorEngine": (".vector", "L3VectorEngine"),
}


def __getattr__(name: str):
    if name in _EXPORT_MAP:
        mod_name, attr_name = _EXPORT_MAP[name]
        mod = import_module(mod_name, __name__)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
