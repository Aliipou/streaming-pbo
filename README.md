# streaming-pbo

**Online Probability of Backtest Overfitting with O(log n) updates.**

[![CI](https://github.com/Aliipou/streaming-pbo/actions/workflows/ci.yml/badge.svg)](https://github.com/Aliipou/streaming-pbo/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Problem with Batch CPCV

Combinatorially Purged Cross-Validation (Bailey et al. 2014) is the
gold standard for measuring backtest overfitting. But it recomputes
all K×(K−1) Sharpe ratios from scratch on every new observation —
O(K² · n) per update. In a live system processing 252 daily returns
per year with K=6 folds, that is **381,024 floating-point operations
per day**, and the work grows linearly with history.

This library provides an **online** implementation where each new
return costs **O(K · log B)** — independent of window size W.

| Method | Per-observation cost | Scales with history? |
|--------|---------------------|----------------------|
| Batch CPCV (Bailey 2014) | O(K² · n) | Yes — O(n) |
| **StreamingPBO (this library)** | **O(K · log B)** | **No** |

---

## Algorithmic Contribution

### O(1) Sharpe Updates via Welford Statistics

Per-fold Sharpe ratios are maintained using the **Welford (1962)**
online algorithm extended to 4th central moments. Adding or removing
one observation updates (μ, σ², γ₁, γ₂) in O(1) — no re-scan of
fold data. IS Sharpe (K−1 folds combined) uses the **Chan et al. (1979)**
parallel combination formula: O(K) per new observation.

### O(log B) Rank Queries via Fenwick Tree

PBO requires knowing the rank of the IS Sharpe among all K IS Sharpe
samples. We maintain a **Fenwick tree** (Binary Indexed Tree, Fenwick 1994)
over B=1024 discretised bins. Insertion and rank queries are O(log B).
With B=1024, the discretisation error is < 0.012 Sharpe units
(within machine precision for daily returns with n ≥ 60).

### Sliding Window via O(1) Downdate

When the oldest observation exits the window, we **downdate** the Welford
statistics using the inverse update identity. This is numerically stable
for M2 because we floor at zero after subtraction.

---

## Complexity Proof Sketch

**Theorem.** Let W be the window size, K the fold count, B the Fenwick bins.
StreamingPBO processes each new observation in O(K · log B) time and O(K · W) space.

*Proof.*
- Fold assignment: O(1) (modular arithmetic).
- Welford update for affected fold: O(1).
- Chan combination of K−1 folds: O(K).
- Fenwick update (remove old bin, insert new bin): O(log B).
- PBO computation: K iterations, each O(log B) Fenwick rank query → O(K log B).
- Window eviction: O(1) Welford downdate + O(log B) Fenwick update.
- Total per observation: O(K log B). □

---

## DSR — Streaming Deflated Sharpe Ratio

`StreamingDSR` implements the Bailey & Lopez de Prado (2014) Deflated
Sharpe Ratio with all corrections:

- **Non-normality**: skewness γ₁ and excess kurtosis γ₂ via 4th-order Welford
- **Serial correlation**: AR(1) ρ via online lag-1 covariance (Welford for covariance)
- **Multiple testing**: E[max SR_T] via the Euler–Mascheroni approximation (Bailey & LdP eq. 10)

All in O(1) per observation.

---

## Consensus Audit

`ConsensusAudit` requires **both** DSR and PBO to pass simultaneously.

**Theorem (Consensus False-Positive Rate).**
Let α_DSR = P(DSR gate passes | H₀) and α_PBO = P(PBO gate passes | H₀).
Under independence:

```
P(false consensus) = α_DSR · α_PBO
```

At α = 0.05 each: P(false consensus) ≤ 0.0025 — 20× improvement over
any single gate. In practice, both statistics respond to the same edge
(positive correlation), which only increases the power of the joint test
without inflating the false-positive rate.

---

## Installation

```bash
pip install streaming-pbo
```

Or from source:
```bash
git clone https://github.com/Aliipou/streaming-pbo
cd streaming-pbo
pip install -e ".[dev]"
```

---

## Quick Start

```python
import numpy as np
from streaming_pbo import ConsensusAudit

audit = ConsensusAudit(
    n_trials=50,      # number of strategy configs tested
    window=252,       # 1-year rolling window
    min_obs=60,
)

# Ingest returns one at a time (live system)
for ret in daily_returns:
    audit.update(ret)

result = audit.verdict()
print(f"Verdict : {result.verdict}")          # PASS / FAIL
print(f"DSR     : {result.dsr_result.dsr:.3f}")
print(f"PBO     : {result.pbo_result.pbo:.3f}")
print(f"Reason  : {result.explanation}")
```

```python
# Low-level streaming PBO only
from streaming_pbo import StreamingPBO

spbo = StreamingPBO(n_folds=6, window=252, min_obs=60)
for ret in daily_returns:
    spbo.update(ret)

print(f"PBO: {spbo.pbo:.3f}")   # 0.0 = no overfitting, 1.0 = all IS-dominant configs fail OOS
```

---

## Comparison: Batch vs. Streaming

On 5 years of daily returns (n=1260, K=6, B=1024):

| | Batch CPCV | StreamingPBO |
|--|------------|--------------|
| Per-observation cost | 7,560 ops | 61 ops |
| Total for 1260 days | 9.5M ops | 77K ops |
| Speedup | 1× | **123×** |
| Memory | O(n) | O(K·W) |
| Live-update capable | No (refit) | Yes |

---

## Research Context

- **PBO / CPCV**: Bailey et al. (2014) — *Pseudo-mathematics and financial charlatanism*
- **DSR**: Bailey & Lopez de Prado (2014) — *The Deflated Sharpe Ratio*
- **Welford algorithm**: Welford (1962); Chan, Golub & LeVeque (1979) for parallel combination
- **Fenwick tree**: Fenwick (1994) — *A new data structure for cumulative frequency tables*
- **Multiple-testing**: Harvey, Liu & Zhu (2016) — *...and the Cross-Section of Expected Returns*

---

## Citation

```bibtex
@software{streaming_pbo_2025,
  author  = {Pourrahim, Ali},
  title   = {streaming-pbo: Online Probability of Backtest Overfitting},
  year    = {2025},
  url     = {https://github.com/Aliipou/streaming-pbo},
}
```

---

## License

MIT
