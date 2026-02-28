# SFE

Independent research

---

## Overview

A self-referential adaptive agent is embedded in a one-dimensional stochastic field. It tracks its own position via a matched Kalman filter, builds a perceived field from its position history, and monitors prediction surprise through a self-regulating windowed gate.

The research investigates structural limits of internal knowing — what a self-referential adaptive system can and cannot detect about its own environment, given its architecture. The agent only has access to its internal representations. The question is how architectural choices shape what becomes knowable, and what becomes structurally invisible, from within.

---

## Core Findings

**SFE-05: The floor-lock and its resolution**

When detection operates on the innovations of a matched adaptive estimator, regime geometry becomes structurally unobservable. The estimator suppresses precisely the statistical structure required for regime discrimination.

This is not a tuning problem. It is a pipeline topology constraint.

The estimator and the detector share a channel. The estimator's objective is to minimize residual variance — and in succeeding, it erases the regime contrast the detector depends on. The observed invariant innovation magnitude (~0.714 under normalization) is a diagnostic symptom of that structural suppression, not the primary result.

The constraint is not specific to Kalman filtering. Any matched predictor that minimizes residual structure will suppress regime contrast in its own error channel.

**Resolution:** detection must be architecturally decoupled from estimation. A null predictor with fixed reference x̂=0 never absorbs regime contrast and preserves geometric separability in position space. Geometric separability restored under burst conditions (3.02σ).

---

**SFE-06: The joint observer manifold**

When no individual observer can access field information, that information is not lost — it is encoded in the geometry of the relationship between observers.

Two null-predictor observers in the same OU field, measuring their windowed cross-correlation, produce a joint state space (x_a, x_b, ρ_ab) whose geometry encodes field confinement strength k in a form inaccessible to either observer individually.

Four results confirmed:

1. **Early detection.** Cross-correlation detects confinement 80 cycles before either single observer crosses its threshold.

2. **Geometric collapse.** The joint manifold transitions from volumetric (e1=0.42) to linear (e1=0.89) under confinement. The collapse axis is entirely in the correlation dimension (|wρ|=1.000).

3. **Relational gap (confirmed).** ρ dominance locks at k≈0.002. The 2D spatial geometry has not collapsed by k=0.030. Minimum confirmed gap: Δk > 0.028. This is not a resolution artifact.

4. **Empirical stability confirmed.** The eigenvalue asymmetry e2/e3 is stable under observer offset δ ∈ {0, 2, 5} (spread < 0.001) and window size W ∈ {20, 40, 80} (spread < 0.003) across all tested configurations. Within this measurement model, W behaves as a lens: varying it does not change the eigenvalue structure. Whether this stability is analytically derivable from the OU covariance structure is the primary open question for SFE-07.

---

## Information Across Representations

The series has traced where field information lives across three levels, each inaccessible from the previous:

| Paper | Where information is | Where it is not |
|---|---|---|
| SFE-05.12b | Position space | Estimator residual |
| SFE-05.13b | Null-ratio channel | Kalman residual |
| SFE-06 | Joint observer manifold | Any single observer |

In each case the information was not destroyed. It relocated to a representation the previous architecture did not instrument.

---

## Version Lineage

**Phase 1 — Architecture construction (05.3–05.7)**

Building the measurement substrate: state manifold, convex hull geometry, coherence detection, adaptive coupling, dual buffer. Establishing that the agent does not hallucinate coherence in pure noise.

| Version | What was built |
|---|---|
| 05.3 | State manifold — convex hull 3D, surprise on Z axis |
| 05.4 | Attractor definition — coherence event recorder, adaptive λ |
| 05.5 | Active inference — coherence centroid as navigation force |
| 05.6 | Dual buffer architecture — epistemic selectivity |
| 05.7 | Detection boundary mapping — drift sweep, zero false positives confirmed |

**Phase 2 — Hitting the wall (05.8–05.13b)**

Each version failed for a distinct architectural reason. Each fix exposed the next layer. The structural constraint only became visible after every local fix had been exhausted.

| Version | Failure | Resolution |
|---|---|---|
| 05.08 | σ_m=0.90 — innovation floor unreachable | Reduce σ_m |
| 05.09 | Kalman periodic resets — spurious transient spikes | Remove resets |
| 05.10 | Gate direction inverted — fires on high ratio | Invert gate logic |
| 05.11 | Reset artefacts persist — delta threshold insufficient | Fix Kalman reset |
| 05.12 | Per-cycle baseline mismatch — ceiling goes negative | Windowed baseline |
| 05.12b | Floor-lock confirmed — zero events (correct result) | Introduce burst phase |
| 05.13 | Kalman floor-lock persists under burst | Decouple channels |
| 05.13b | Null predictor — burst detected ✓ | Resolved |

The lineage is the argument. The fact that no parameter adjustment resolved the failure — only an architectural change did — is what makes the result structural.

**Phase 3 — The joint manifold (SFE-06 through 06.7)**

Having established what a single observer cannot see, the series moves to what becomes visible in the relationship between observers. Each sub-version answered a specific structural question.

| Version | Question answered |
|---|---|
| SFE-06 | Two-observer cross-correlation detects 80 cycles earlier |
| SFE-06.2 | Independent RNG confirmed; volumetric/planar/linear classification established |
| SFE-06.3 | Collapse axis confirmed purely relational (|wρ|=1.000) |
| SFE-06.4 | Full manifold mapped across k ∈ [0, 5]; phase transition at k≈0.1 |
| SFE-06.5 | Transition boundary resolved: two-step (VOLUMETRIC→PLANAR→LINEAR) |
| SFE-06.6 | Relational gap confirmed (Δk > 0.028); δ-invariance confirmed |
| SFE-06.7 | W is a lens: e2/e3 invariant across W ∈ {20, 40, 80}; f(k) isolated |

---

## Status

| Paper | Status | DOI | GitHub Mirror |
|---|---|---|
| SFE-05.12b / 05.13b | Confirmed ✓ — frozen | [10.5281/zenodo.18808974](https://doi.org/10.5281/zenodo.18808974) | [Mirror](doc/sfe_v8.pdf) |
| SFE-06 (v3, through 06.7) | Confirmed ✓ — ready to freeze | — | [Mirror](doc/sfe_06_v3.pdf) |
| SFE-07 | Open — derive or falsify f(k) from OU covariance | — | — |

---

## Open Direction: SFE-07

SFE-06.7 confirmed that e2/e3 is empirically stable across all tested configurations of δ and W, depending on k alone within the tested range.

The analytical question is now isolated: is this stability derivable from the OU steady-state covariance matrix for two particles in the same harmonic well, including the windowed cross-correlation term? Or does it break under conditions not yet tested?

Deriving f(k) — or finding where it fails — is the first result in the SFE series that requires writing down the math the geometry has been pointing toward. It is also the first genuine test of whether the empirical stability observed is a structural property or a coincidence of the tested parameter range.

That is SFE-07.

---

*February 2026*