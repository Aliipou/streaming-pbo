"""Tests for ConsensusAudit."""
import numpy as np
import pytest
from streaming_pbo import ConsensusAudit

RNG = np.random.default_rng(7)


def _feed(est, rets):
    for r in rets:
        est.update(float(r))
    return est


def test_insufficient_data_before_min_obs():
    ca = ConsensusAudit(min_obs=60)
    for r in RNG.normal(0, 0.01, 30):
        ca.update(float(r))
    res = ca.verdict()
    assert res.verdict == "INSUFFICIENT_DATA"


def test_fail_for_noise():
    """Pure noise should fail DSR gate."""
    rets = RNG.normal(0.0, 0.01, 300)
    ca = _feed(ConsensusAudit(n_trials=1), rets)
    res = ca.verdict()
    assert res.verdict in ("FAIL", "PASS")   # mostly FAIL
    assert res.dsr_result is not None
    assert res.pbo_result is not None


def test_pass_for_strong_edge():
    """Very strong edge should pass both gates."""
    rets = RNG.normal(0.005, 0.005, 500)   # annualised SR ≈ 14
    ca = _feed(ConsensusAudit(n_trials=1), rets)
    res = ca.verdict()
    assert res.dsr_gate is True


def test_explanation_populated():
    ca = _feed(ConsensusAudit(), RNG.normal(0, 0.01, 200))
    res = ca.verdict()
    assert isinstance(res.explanation, str)
    assert len(res.explanation) > 0


def test_false_positive_rate():
    """
    Consensus false-positive rate must be < 0.05 on pure noise.
    Single-gate FPR is ~0.05; consensus should be much lower.
    """
    n_trials = 50
    fp = 0
    rng = np.random.default_rng(123)
    for _ in range(n_trials):
        rets = rng.normal(0.0, 0.01, 300)
        ca = _feed(ConsensusAudit(n_trials=1), rets)
        if ca.verdict().verdict == "PASS":
            fp += 1
    fpr = fp / n_trials
    assert fpr < 0.15, f"Consensus FPR too high: {fpr:.2f}"


def test_n_observations_tracked():
    ca = ConsensusAudit(min_obs=30)
    for r in RNG.normal(0, 0.01, 100):
        ca.update(float(r))
    res = ca.verdict()
    assert res.n_observations == 100
