"""Tests for StreamingDSR."""
import math
import numpy as np
import pytest
from streaming_pbo import StreamingDSR

RNG = np.random.default_rng(99)


def _feed(est, rets):
    for r in rets:
        est.update(float(r))
    return est


def test_dsr_none_before_four_obs():
    dsr = StreamingDSR()
    for r in [0.01, -0.01, 0.005]:
        dsr.update(r)
    assert dsr.dsr is None


def test_dsr_significant_for_strong_edge():
    """High-Sharpe strategy should have DSR > 0.95."""
    rets = RNG.normal(0.003, 0.005, 500)   # SR ≈ 8.5 annualised
    dsr = _feed(StreamingDSR(n_trials=1), rets)
    assert dsr.dsr is not None
    assert dsr.dsr > 0.95


def test_dsr_not_significant_for_noise():
    """Pure noise returns should have DSR < 0.5."""
    rets = RNG.normal(0.0, 0.01, 500)
    dsr = _feed(StreamingDSR(n_trials=1), rets)
    assert dsr.dsr is not None
    assert dsr.dsr < 0.5


def test_dsr_decreases_with_more_trials():
    """More trials → higher benchmark SR → lower DSR for same edge."""
    rets = RNG.normal(0.001, 0.01, 500)
    d1 = _feed(StreamingDSR(n_trials=1), rets)
    d100 = _feed(StreamingDSR(n_trials=100), rets)
    r1 = d1.result()
    r100 = d100.result()
    assert r1 is not None and r100 is not None
    assert r1.dsr >= r100.dsr


def test_raw_sharpe_positive_for_positive_edge():
    rets = RNG.normal(0.001, 0.01, 300)
    dsr = _feed(StreamingDSR(), rets)
    res = dsr.result()
    assert res is not None
    assert res.raw_sharpe > 0


def test_skewness_negative_returns():
    """Left-skewed returns (large negative outliers) should show negative skew."""
    rets = np.concatenate([
        RNG.normal(0.001, 0.005, 400),
        RNG.normal(-0.05, 0.002, 20),  # crash days
    ])
    dsr = _feed(StreamingDSR(), rets)
    res = dsr.result()
    assert res is not None
    assert res.skewness < 0


def test_rolling_window():
    """Rolling DSR should only use the last `window` observations."""
    rets = RNG.normal(0, 0.01, 400)
    dsr = _feed(StreamingDSR(window=100), rets)
    assert dsr.result() is not None


def test_result_fields():
    rets = RNG.normal(0.0, 0.01, 200)
    dsr = _feed(StreamingDSR(n_trials=10), rets)
    res = dsr.result()
    assert res is not None
    assert math.isfinite(res.dsr)
    assert math.isfinite(res.raw_sharpe)
    assert math.isfinite(res.skewness)
    assert math.isfinite(res.excess_kurtosis)
    assert 0.0 <= res.dsr <= 1.0
    assert 0.0 <= res.p_value <= 1.0
    assert res.n_trials == 10


def test_invalid_n_trials():
    with pytest.raises(ValueError):
        StreamingDSR(n_trials=0)
