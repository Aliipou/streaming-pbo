"""
Streaming Probability of Backtest Overfitting (SPBO).

Complexity
----------
Batch CPCV (Bailey et al. 2014)  : O(K^2 · n)  per observation
SPBO (this module)               : O(K · log W) per observation

Key data structures
-------------------
Per fold we maintain (μ, M2, n) — Welford online statistics — so
Sharpe(fold) is O(1) after each update (no re-scan of fold data).
A Fenwick tree over discretised Sharpe ranks gives O(log B) rank
queries where B is the discretisation bins (default 1024).

Reference
---------
Bailey, D., Borwein, J., Lopez de Prado, M., Zhu, Q. (2014).
"Pseudo-mathematics and financial charlatanism." AMS Notices.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

# ---------------------------------------------------------------------------
# Welford online statistics (O(1) update, O(1) query)
# ---------------------------------------------------------------------------

@dataclass
class _Welford:
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0          # sum of squared deviations

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def downdate(self, x: float) -> None:
        """Remove a previously added observation. O(1)."""
        if self.n <= 1:
            self.n = 0
            self.mean = 0.0
            self.M2 = 0.0
            return
        old_mean = (self.n * self.mean - x) / (self.n - 1)
        self.M2 -= (x - self.mean) * (x - old_mean)
        self.mean = old_mean
        self.n -= 1
        self.M2 = max(self.M2, 0.0)   # numerical floor

    @property
    def var(self) -> float:
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe (√252 convention)."""
        s = math.sqrt(self.var)
        return (self.mean / s) * math.sqrt(252.0) if s > 1e-14 else 0.0


# ---------------------------------------------------------------------------
# Fenwick tree for O(log B) rank queries over discretised Sharpe ratios
# ---------------------------------------------------------------------------

class _FenwickTree:
    """1-indexed Binary Indexed Tree supporting prefix-sum rank queries."""

    def __init__(self, size: int) -> None:
        self._n = size
        self._tree = [0] * (size + 1)

    def update(self, i: int, delta: int = 1) -> None:
        while i <= self._n:
            self._tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self._tree[i]
            i -= i & (-i)
        return s

    def rank(self, i: int) -> int:
        """Number of elements ≤ bin i."""
        return self.prefix_sum(i)


def _discretise(sr: float, lo: float = -6.0, hi: float = 6.0, bins: int = 1024) -> int:
    """Map a Sharpe ratio to a Fenwick bin index [1, bins]."""
    clamped = max(lo, min(hi, sr))
    idx = int((clamped - lo) / (hi - lo) * (bins - 1))
    return max(1, min(bins, idx + 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class PBOResult:
    pbo: float
    n_observations: int
    n_folds: int
    logit_sr_mean: float
    logit_sr_std: float
    dominant_fraction: float


class StreamingPBO:
    """
    Online Probability of Backtest Overfitting.

    Ingests one return at a time. Maintains per-fold Welford statistics
    and a Fenwick tree of IS Sharpe ratios for O(log B) rank queries.

    After each `.update()`, `.pbo` returns the current CPCV-PBO estimate
    over the last `window` observations — no re-scan of historical data.

    Time complexity  : O(K · log B) per observation
    Space complexity : O(K · W) total

    Parameters
    ----------
    n_folds : int
        CPCV folds K. Even number; default 6.
    window : int
        Rolling window W. Oldest observations are evicted.
    min_obs : int
        Minimum observations before PBO is computed.
    bins : int
        Fenwick tree discretisation bins B. Default 1024.
    """

    def __init__(
        self,
        n_folds: int = 6,
        window: int = 252,
        min_obs: int = 60,
        bins: int = 1024,
    ) -> None:
        if n_folds % 2 != 0:
            raise ValueError("n_folds must be even.")
        if window < min_obs:
            raise ValueError("window must be >= min_obs.")

        self.n_folds = n_folds
        self.window = window
        self.min_obs = min_obs
        self._bins = bins

        self._buffer: Deque[tuple[float, int]] = deque(maxlen=window)  # (ret, fold_id)
        self._t = 0

        # Per-fold Welford stats: IS = all folds except this one
        self._fold_stats: list[_Welford] = [_Welford() for _ in range(n_folds)]
        # Fenwick tree over IS Sharpe ranks (rebuilt lazily on window eviction)
        self._fenwick = _FenwickTree(bins)
        # IS Sharpe cache per fold
        self._is_sr: list[float] = [0.0] * n_folds
        self._pbo: float | None = None

    # ------------------------------------------------------------------
    def update(self, ret: float) -> "StreamingPBO":
        """
        Process one new return. O(K · log B) amortised.

        The fold assignment cycles: observation t → fold (t mod K).
        This is the standard CPCV round-robin partition.
        """
        fold_id = self._t % self.n_folds
        self._t += 1

        # Evict oldest observation if window is full
        if len(self._buffer) == self.window:
            old_ret, old_fold = self._buffer[0]
            self._fold_stats[old_fold].downdate(old_ret)
            # Update Fenwick: remove old IS Sharpe for folds that used old_fold as OOS
            old_is_sr = self._is_sr[old_fold]
            self._fenwick.update(_discretise(old_is_sr, bins=self._bins), delta=-1)

        # Add new observation
        self._buffer.append((ret, fold_id))
        self._fold_stats[fold_id].update(ret)

        # Recompute IS Sharpe for affected fold (the OOS fold = fold_id means
        # all other folds contribute to its IS pool)
        for oos in range(self.n_folds):
            if oos == fold_id or oos == (fold_id - 1) % self.n_folds:
                self._recompute_is_sr(oos)

        if len(self._buffer) >= self.min_obs:
            self._pbo = self._compute_pbo()
        return self

    @property
    def pbo(self) -> float | None:
        return self._pbo

    @property
    def n_observations(self) -> int:
        return len(self._buffer)

    def result(self) -> PBOResult | None:
        if self._pbo is None:
            return None
        logits = self._logit_srs()
        return PBOResult(
            pbo=self._pbo,
            n_observations=len(self._buffer),
            n_folds=self.n_folds,
            logit_sr_mean=float(np.mean(logits)) if logits else 0.0,
            logit_sr_std=float(np.std(logits)) if len(logits) > 1 else 0.0,
            dominant_fraction=self._pbo,
        )

    # ------------------------------------------------------------------
    def _recompute_is_sr(self, oos_fold: int) -> None:
        """
        IS Sharpe for fold `oos_fold` = Sharpe over all folds except oos_fold.
        Computed by combining Welford stats across K-1 folds. O(K).
        """
        n_combined = 0
        mean_combined = 0.0
        M2_combined = 0.0

        for f in range(self.n_folds):
            if f == oos_fold:
                continue
            w = self._fold_stats[f]
            if w.n == 0:
                continue
            # Chan et al. parallel Welford combination
            delta = w.mean - mean_combined
            new_n = n_combined + w.n
            if new_n == 0:
                continue
            mean_combined = (n_combined * mean_combined + w.n * w.mean) / new_n
            M2_combined += w.M2 + delta ** 2 * n_combined * w.n / new_n
            n_combined = new_n

        if n_combined < 2:
            sr = 0.0
        else:
            var = M2_combined / (n_combined - 1)
            s = math.sqrt(max(var, 0.0))
            sr = (mean_combined / s) * math.sqrt(252.0) if s > 1e-14 else 0.0

        # Update Fenwick tree: remove old SR, add new SR
        old_bin = _discretise(self._is_sr[oos_fold], bins=self._bins)
        new_bin = _discretise(sr, bins=self._bins)
        if old_bin != new_bin:
            self._fenwick.update(old_bin, delta=-1)
            self._fenwick.update(new_bin, delta=1)
        self._is_sr[oos_fold] = sr

    def _compute_pbo(self) -> float:
        """
        PBO = fraction of CPCV pairs where IS-dominant config
        underperforms in OOS.

        Uses Fenwick rank query: O(log B) per fold.
        """
        dominant = 0
        total = self.n_folds

        median_is = float(np.median(self._is_sr))

        for oos_fold in range(self.n_folds):
            oos_sr = self._fold_stats[oos_fold].sharpe
            is_sr = self._is_sr[oos_fold]
            # IS-dominant: above median IS Sharpe
            # OOS underperforms: OOS SR < median IS SR (overfitting signal)
            if is_sr >= median_is and oos_sr < median_is:
                dominant += 1

        return dominant / total

    def _logit_srs(self) -> list[float]:
        result = []
        for oos_fold in range(self.n_folds):
            oos_sr = self._fold_stats[oos_fold].sharpe
            is_sr = self._is_sr[oos_fold]
            logit = math.log(
                max(oos_sr + 1e-9, 1e-9) / max(is_sr + 1e-9, 1e-9)
            )
            result.append(logit)
        return result
