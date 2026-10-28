# Boundary-transmission Lean proof

Goal: formalize the existing paper's mathematics in a pinned Lean 4/mathlib project.
The empirical data and the Python simulator are outside the proof boundary.

- Establish Lean 4.19.0/mathlib v4.19.0 and a reproducible Lake manifest.
- Prove finite-census decomposition, descendant-count covariance, telescoping,
  survivor cancellation, and the newborn-endpoint identity from finite sums.
- Prove positive expectation for bounded nonnegative innovation using probability
  measures. Investigate the exact clipped-Gaussian integral without assuming the
  result being claimed; disclose any remaining coverage gap.
- Compile every theorem; prohibit placeholders and new axioms. Audit theorem axiom
  dependencies and explicitly require nonempty censuses where means are interpreted.
- Document theorem-to-paper mapping and commands. Add a short reference in the
  existing .tex only if useful; use its built-in compiler and leave that editor open.
- Commit the proof supplement without changing raw experiment evidence or verdicts.

This is formal verification of established mathematics, not a new experiment or
confirmation of publication novelty. The user has already authorized implementation.
