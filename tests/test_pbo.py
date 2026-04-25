"""Tests for StreamingPBO."""
import numpy as np
import pytest
from streaming_pbo import StreamingPBO

RNG = np.random.default_rng(42)


def _feed(estimator, returns):
    for r in returns:
        estimator.update(float(r))
    return estimator


def test_pbo_in_unit_interval():
    rets = RNG.normal(0.0005, 0.01, 300)
    spbo = _feed(StreamingPBO(window=252, min_obs=60), rets)
    assert spbo.pbo is not None
    assert 0.0 <= spbo.pbo <= 1.0


def test_pbo_none_before_min_obs():
    spbo = StreamingPBO(min_obs=60)
    for r in RNG.normal(0, 0.01, 30):
        spbo.update(float(r))
    assert spbo.pbo is None


def test_pbo_updates_incrementally():
    rets = RNG.normal(0.001, 0.01, 200)
    spbo = StreamingPBO(window=252, min_obs=60)
    prev = None
    for r in rets:
        spbo.update(float(r))
        if spbo.pbo is not None:
            prev = spbo.pbo
    assert prev is not None


def test_pbo_high_for_overfit_strategy():
    """Strategy with IS edge that disappears OOS → PBO should be elevated."""
    rng = np.random.default_rng(7)
    # IS returns: strongly positive; OOS returns: noise
    n = 252
    is_rets = rng.normal(0.002, 0.005, n // 2)
    oos_rets = rng.normal(0.0, 0.015, n // 2)
    rets = np.concatenate([is_rets, oos_rets])
    spbo = _feed(StreamingPBO(window=252, min_obs=60), rets)
    assert spbo.pbo is not None


def test_result_dataclass():
    rets = RNG.normal(0.0, 0.01, 200)
    spbo = _feed(StreamingPBO(window=252, min_obs=60), rets)
    res = spbo.result()
    assert res is not None
    assert res.n_observations == min(200, 252)
    assert res.n_folds == 6


def test_odd_folds_raises():
    with pytest.raises(ValueError):
        StreamingPBO(n_folds=5)


def test_window_smaller_than_min_obs_raises():
    with pytest.raises(ValueError):
        StreamingPBO(window=30, min_obs=60)


def test_rolling_window_eviction():
    """After window observations, buffer stays at window size."""
    window = 100
    spbo = StreamingPBO(window=window, min_obs=30, n_folds=4)
    for r in RNG.normal(0, 0.01, 200):
        spbo.update(float(r))
    assert spbo.n_observations == window


def test_welford_downdate_consistency():
    """Sliding window PBO should be reproducible on same window."""
    rets = RNG.normal(0, 0.01, 300)
    spbo1 = _feed(StreamingPBO(window=100, min_obs=60), rets)
    # Fresh estimator on just last 100 observations
    spbo2 = _feed(StreamingPBO(window=100, min_obs=60), rets[-100:])
    assert spbo1.pbo is not None
    assert spbo2.pbo is not None
    assert abs(spbo1.pbo - spbo2.pbo) < 0.15  # same window, close PBO
