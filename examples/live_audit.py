"""
Live strategy audit example.

Simulates two strategies observed in real-time:
  - Strategy A: genuine edge (μ > 0, consistent)
  - Strategy B: overfit (IS edge disappears OOS)

Both are audited via ConsensusAudit as returns arrive.
"""

import numpy as np
from streaming_pbo import ConsensusAudit

rng = np.random.default_rng(0)

# Strategy A: genuine edge
returns_a = rng.normal(0.0015, 0.008, 400)

# Strategy B: in-sample edge that reverses out-of-sample
returns_b = np.concatenate([
    rng.normal(0.0020, 0.006, 200),   # IS phase: looks great
    rng.normal(-0.0005, 0.012, 200),  # OOS phase: edge gone
])

audit_a = ConsensusAudit(n_trials=50, window=252, min_obs=60)
audit_b = ConsensusAudit(n_trials=50, window=252, min_obs=60)

print(f"{'Day':>4}  {'A verdict':>16}  {'A DSR':>7}  {'A PBO':>7}  {'B verdict':>16}  {'B DSR':>7}  {'B PBO':>7}")
print("-" * 80)

for t in range(400):
    audit_a.update(float(returns_a[t]))
    audit_b.update(float(returns_b[t]))

    if t % 50 == 49:
        ra = audit_a.verdict()
        rb = audit_b.verdict()

        dsr_a = f"{ra.dsr_result.dsr:.3f}" if ra.dsr_result else "—"
        pbo_a = f"{ra.pbo_result.pbo:.3f}" if ra.pbo_result else "—"
        dsr_b = f"{rb.dsr_result.dsr:.3f}" if rb.dsr_result else "—"
        pbo_b = f"{rb.pbo_result.pbo:.3f}" if rb.pbo_result else "—"

        print(f"{t+1:>4}  {ra.verdict:>16}  {dsr_a:>7}  {pbo_a:>7}  {rb.verdict:>16}  {dsr_b:>7}  {pbo_b:>7}")
