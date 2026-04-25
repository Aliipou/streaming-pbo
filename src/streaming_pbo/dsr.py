"""
Streaming Deflated Sharpe Ratio (DSR).

Corrects the observed Sharpe for:
  1. Non-normality (skewness γ₁, excess kurtosis γ₂)
  2. Serial correlation (via AR(1) autocorrelation ρ)
  3. Multiple testing (number of independent trials T)

Reference
---------
Bailey, D. & Lopez de Prado, M. (2014).
"The Deflated Sharpe Ratio: Correcting for Selection Bias,
Backtest Overfitting, and Non-Normality." JPortfolio Management.

Formula (Bailey & LdP 2014, eq. 12)
-------------------------------------
DSR = SR* · √(T-1) · N(SR* · √(T-1) · ê)   — see code for full expression

where SR* = (SR - E[SR_max]) / Var[SR_max]^(1/2)
and ê = correction factor from skewness, kurtosis, autocorrelation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque

from scipy.special import ndtr  # standard normal CDF, avoids scipy.stats overhead


@dataclass
class DSRResult:
    dsr: float
    raw_sharpe: float
    benchmark_sharpe: float
    skewness: float
    excess_kurtosis: float
    autocorr_ar1: float
    n_trials: int
    n_observations: int
    is_significant: bool        # DSR > 0
    p_value: float


class StreamingDSR:
    """
    Online Deflated Sharpe Ratio.

    Maintains Welford online statistics for mean, variance, skewness,
    and kurtosis (4th-order online algorithm). AR(1) autocorrelation
    is estimated via a lag-1 running covariance.

    All updates are O(1). The DSR is O(1) to query.

    Parameters
    ----------
    n_trials : int
        Number of strategy configurations tested. Used for multiple-
        testing correction via the expected maximum Sharpe formula.
    window : int | None
        Rolling window. None = expanding window.
    benchmark_sr : float
        Sharpe ratio of the null strategy. Default 0.0.
    annualize : float
        Annualisation factor. Default 252 (daily returns).
    """

    def __init__(
        self,
        n_trials: int = 1,
        window: int | None = None,
        benchmark_sr: float = 0.0,
        annualize: float = 252.0,
    ) -> None:
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1.")

        self.n_trials = n_trials
        self.window = window
        self.benchmark_sr = benchmark_sr
        self.annualize = annualize

        self._buffer: Deque[float] | None = deque(maxlen=window) if window else None
        self._t = 0

        # Welford up to 4th central moment (Terriberry 2007 extension)
        self._n = 0
        self._M1 = 0.0   # mean
        self._M2 = 0.0   # sum of (x - mean)^2
        self._M3 = 0.0
        self._M4 = 0.0

        # AR(1) lag-1 covariance: running Cov(r_t, r_{t-1})
        self._prev: float | None = None
        self._lag_cov_n = 0
        self._lag_mean_x = 0.0
        self._lag_mean_y = 0.0
        self._lag_C = 0.0        # running covariance sum (Welford for cov)

    # ------------------------------------------------------------------
    def update(self, ret: float) -> "StreamingDSR":
        """Ingest one return. O(1)."""
        if self._buffer is not None:
            if len(self._buffer) == self.window:
                old = self._buffer[0]
                self._downdate(old)
                if self._prev is not None:
                    self._lag_downdate(old, self._prev)
            self._buffer.append(ret)

        # 4-th order Welford update
        n1 = self._n
        self._n += 1
        n = self._n
        delta = ret - self._M1
        delta_n = delta / n
        delta_n2 = delta_n ** 2
        term1 = delta * delta_n * n1
        self._M1 += delta_n
        self._M4 += (
            term1 * delta_n2 * (n * n - 3 * n + 3)
            + 6 * delta_n2 * self._M2
            - 4 * delta_n * self._M3
        )
        self._M3 += term1 * delta_n * (n - 2) - 3 * delta_n * self._M2
        self._M2 += term1

        # AR(1) lag-1 covariance update
        if self._prev is not None:
            self._lag_cov_n += 1
            k = self._lag_cov_n
            dx = self._prev - self._lag_mean_x
            dy = ret - self._lag_mean_y
            self._lag_mean_x += dx / k
            self._lag_mean_y += dy / k
            self._lag_C += dx * (ret - self._lag_mean_y)
        self._prev = ret
        self._t += 1
        return self

    @property
    def dsr(self) -> float | None:
        if self._n < 4:
            return None
        return self._compute_dsr().dsr

    def result(self) -> DSRResult | None:
        if self._n < 4:
            return None
        return self._compute_dsr()

    # ------------------------------------------------------------------
    def _downdate(self, x: float) -> None:
        """Remove oldest observation from 4th-order Welford. O(1)."""
        if self._n <= 1:
            self._n = 0
            self._M1 = self._M2 = self._M3 = self._M4 = 0.0
            return
        n = self._n
        delta = x - self._M1
        delta_n = delta / (n - 1)
        self._M4 -= (
            delta * delta_n ** 2 * (n * n - 3 * n + 3)
            + 6 * delta_n ** 2 * self._M2
            - 4 * delta_n * self._M3
        )
        self._M3 -= delta * delta_n * (n - 2) * delta_n - 3 * delta_n * self._M2
        self._M2 -= delta * (x - (self._M1 * n - x) / (n - 1))
        self._M1 = (self._M1 * n - x) / (n - 1)
        self._M2 = max(self._M2, 0.0)
        self._n -= 1

    def _lag_downdate(self, old_x: float, old_y: float) -> None:
        if self._lag_cov_n <= 0:
            return
        k = self._lag_cov_n
        self._lag_C -= (old_x - self._lag_mean_x) * (old_y - self._lag_mean_y)
        self._lag_mean_x = (self._lag_mean_x * k - old_x) / max(k - 1, 1)
        self._lag_mean_y = (self._lag_mean_y * k - old_y) / max(k - 1, 1)
        self._lag_cov_n = max(self._lag_cov_n - 1, 0)

    def _compute_dsr(self) -> DSRResult:
        n = self._n
        var = self._M2 / (n - 1)
        std = math.sqrt(max(var, 1e-28))

        raw_sr = (self._M1 / std) * math.sqrt(self.annualize)

        skew = (self._M3 / n) / (self._M2 / n) ** 1.5 if self._M2 > 0 else 0.0
        kurt = (self._M4 / n) / (self._M2 / n) ** 2 - 3.0 if self._M2 > 0 else 0.0

        # AR(1) autocorrelation
        lag_var_x = self._lag_C / max(self._lag_cov_n - 1, 1)
        rho = lag_var_x / max(var, 1e-28)
        rho = max(-0.999, min(0.999, rho))

        # Effective observations after autocorrelation adjustment (Lo 2002)
        n_eff = n * (1 - rho) / (1 + rho) if abs(rho) < 1 else 1.0
        n_eff = max(n_eff, 2.0)

        # SR standard deviation including higher moments (Bailey & LdP eq. 6)
        sr_annual = raw_sr
        sr_daily = sr_annual / math.sqrt(self.annualize)
        sr_std = math.sqrt(
            (1 - skew * sr_daily + (kurt / 4) * sr_daily ** 2) / (n_eff - 1)
        ) * math.sqrt(self.annualize)

        # Expected maximum Sharpe under multiple testing (Bailey & LdP eq. 10)
        T = self.n_trials
        if T > 1:
            e_max = _expected_max_sr(T, sr_std, n_eff)
        else:
            e_max = self.benchmark_sr

        # DSR: t-statistic against expected maximum (Bailey & LdP eq. 12)
        if sr_std < 1e-14:
            dsr_val = 0.0
            p_val = 0.5
        else:
            z = (raw_sr - e_max) / sr_std
            dsr_val = float(ndtr(z))
            p_val = 1.0 - dsr_val

        return DSRResult(
            dsr=dsr_val,
            raw_sharpe=raw_sr,
            benchmark_sharpe=e_max,
            skewness=skew,
            excess_kurtosis=kurt,
            autocorr_ar1=rho,
            n_trials=T,
            n_observations=n,
            is_significant=dsr_val > 0.95,
            p_value=p_val,
        )


def _expected_max_sr(T: int, sr_std: float, n: float) -> float:
    """
    Expected maximum of T i.i.d. Sharpe ratios.

    Approximation from Bailey & Lopez de Prado (2014, eq. 10):
      E[max SR_T] ≈ (1 - γ) · Φ^{-1}(1 - 1/T) + γ · Φ^{-1}(1 - 1/(T·e))

    where γ = Euler-Mascheroni constant ≈ 0.5772.
    """
    from scipy.special import ndtri  # inverse normal CDF

    gamma_em = 0.5772156649
    if T <= 1:
        return 0.0
    z1 = float(ndtri(1.0 - 1.0 / T))
    z2 = float(ndtri(1.0 - 1.0 / (T * math.e)))
    return sr_std * ((1 - gamma_em) * z1 + gamma_em * z2)
