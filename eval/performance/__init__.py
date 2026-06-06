"""Three-pillar TST performance evaluation.

Public surface (unchanged from the former single-module ``performance.py``):
``from tst_utils.eval.performance import TstPerformanceMetrics, TARGET_STYLES``.
"""

from tst_utils.eval.performance.constants import TARGET_STYLES
from tst_utils.eval.performance.core import TstPerformanceMetrics

__all__ = ["TstPerformanceMetrics", "TARGET_STYLES"]
