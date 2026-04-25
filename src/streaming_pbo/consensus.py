"""
Consensus Audit: joint DSR + PBO gate.

Theory
------
Let T_DSR = 1 if DSR > 0.95, T_PBO = 1 if PBO < 0.55.
Under independence (conservative assumption):
  P(false positive) = P(T_DSR=1 | H₀) · P(T_PBO=1 | H₀) ≤ α_DSR · α_PBO

At α = 0.05 each: P(false consensus) ≤ 0.0025 — an order of magnitude
improvement over any single gate.

In practice T_DSR and T_PBO are positively correlated (both respond
to edge), which only tightens the bound relative to independence.
"""

from __future__ import annotations

from dataclasses import dataclass

from streaming_pbo.dsr import StreamingDSR, DSRResult
from streaming_pbo.pbo import StreamingPBO, PBOResult


@dataclass
class ConsensusResult:
    verdict: str                   # "PASS" | "FAIL" | "INSUFFICIENT_DATA"
    dsr_result: DSRResult | None
    pbo_result: PBOResult | None
    dsr_gate: bool
    pbo_gate: bool
    n_observations: int
    explanation: str


class ConsensusAudit:
    """
    Live consensus audit combining streaming DSR and streaming PBO.

    Both gates must pass for a strategy to be flagged as having
    a genuine edge. This reduces false-positive rate from O(α) to O(α²).

    Parameters
    ----------
    n_trials : int
        Number of strategy configurations tested (for DSR correction).
    dsr_threshold : float
        DSR significance threshold. Default 0.95.
    pbo_threshold : float
        PBO must be below this value. Default 0.55.
    window : int
        Rolling window for both sub-estimators.
    n_folds : int
        CPCV folds for PBO.
    min_obs : int
        Minimum observations before issuing a verdict.
    """

    def __init__(
        self,
        n_trials: int = 1,
        dsr_threshold: float = 0.95,
        pbo_threshold: float = 0.55,
        window: int = 252,
        n_folds: int = 6,
        min_obs: int = 60,
    ) -> None:
        self.dsr_threshold = dsr_threshold
        self.pbo_threshold = pbo_threshold
        self.min_obs = min_obs

        self._dsr = StreamingDSR(n_trials=n_trials, window=window)
        self._pbo = StreamingPBO(n_folds=n_folds, window=window, min_obs=min_obs)

    def update(self, ret: float) -> "ConsensusAudit":
        self._dsr.update(ret)
        self._pbo.update(ret)
        return self

    def verdict(self) -> ConsensusResult:
        dsr_res = self._dsr.result()
        pbo_res = self._pbo.result()

        if dsr_res is None or pbo_res is None:
            return ConsensusResult(
                verdict="INSUFFICIENT_DATA",
                dsr_result=dsr_res,
                pbo_result=pbo_res,
                dsr_gate=False,
                pbo_gate=False,
                n_observations=self._pbo.n_observations,
                explanation=f"Need {self.min_obs} observations; have {self._pbo.n_observations}.",
            )

        dsr_gate = dsr_res.dsr >= self.dsr_threshold
        pbo_gate = (pbo_res.pbo < self.pbo_threshold)
        pass_ = dsr_gate and pbo_gate

        parts = []
        if not dsr_gate:
            parts.append(f"DSR={dsr_res.dsr:.3f} < {self.dsr_threshold} (not significant)")
        if not pbo_gate:
            parts.append(f"PBO={pbo_res.pbo:.3f} >= {self.pbo_threshold} (overfitting likely)")

        explanation = (
            "Both gates passed — genuine edge supported."
            if pass_
            else "Failed: " + "; ".join(parts)
        )

        return ConsensusResult(
            verdict="PASS" if pass_ else "FAIL",
            dsr_result=dsr_res,
            pbo_result=pbo_res,
            dsr_gate=dsr_gate,
            pbo_gate=pbo_gate,
            n_observations=self._pbo.n_observations,
            explanation=explanation,
        )
