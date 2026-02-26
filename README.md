# SFE

Independent research

---

## Overview

A self-referential adaptive agent is embedded in a one-dimensional stochastic field. It tracks its own position via a matched Kalman filter, builds a perceived field from its position history, and monitors prediction surprise through a self-regulating windowed gate.

The research investigates structural limits of internal knowing — what a self-referential adaptive system can and cannot detect about its own environment, given its architecture. The agent only has access to its internal representations. The question is how architectural choices shape what becomes knowable, and what becomes structurally invisible, from within.

---

## Core Finding

When detection operates on the innovations of a matched adaptive estimator, regime geometry becomes structurally unobservable. The estimator suppresses precisely the statistical structure required for regime discrimination.

This is not a tuning problem. It is a pipeline topology constraint.

The estimator and the detector share a channel. The estimator's objective is to minimize residual variance — and in succeeding, it erases the regime contrast the detector depends on. The observed invariant innovation magnitude (~0.714 under normalization) is a diagnostic symptom of that structural suppression, not the primary result.

The constraint is not specific to Kalman filtering. Any matched predictor that minimizes residual structure will suppress regime contrast in its own error channel.

**Resolution:** detection must be architecturally decoupled from estimation. A null predictor with fixed reference x̂=0 never absorbs regime contrast and preserves geometric separability in position space. Geometric separability restored under burst conditions (3.02σ in evaluation).

v8 pdf: [`doc/sfe_v8.pdf`](doc/sfe_v8.pdf)

---

## Version Lineage

The series has two distinct phases.

**Phase 1 — Architecture construction (05.3–05.7)**
Building the measurement substrate: state manifold, convex hull geometry, coherence detection, adaptive coupling, dual buffer. Establishing that the agent does not hallucinate coherence in pure noise. Identifying the detection boundary under minimal field structure.

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

---

## Status

| Version | Status |
|---|---|
| 05.12b | Floor-lock confirmed ✓ |
| 05.13b | Architectural resolution confirmed ✓ |
| 06 | Multi-observer extension — open |

---

*February 2026*