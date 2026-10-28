import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-!
# Positive innovation from a lower trait boundary

This file formalizes the manuscript's elementary boundary-innovation statement
using Mathlib's measure-theoretic Bochner integral. For a probability measure,
this integral is the expectation of a real-valued random variable.

The boundedness and measurability hypotheses establish integrability rather
than assuming it silently. This matters because Mathlib defines the Bochner
integral of a nonintegrable function to be zero.

No particular mutation distribution, Gaussian formula, simulator, or empirical
result is assumed or verified here.
-/

open MeasureTheory
open scoped ENNReal

namespace BoundaryTransmission

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable {X : Ω → ℝ} {U : ℝ}

/-- A measurable trait bounded almost surely in `[0,U]` is integrable on a
finite measure space. A probability measure is a special case. -/
theorem integrable_of_ae_bounded [IsFiniteMeasure μ]
    (hX : Measurable X)
    (hbounded : ∀ᵐ ω ∂μ, 0 ≤ X ω ∧ X ω ≤ U) :
    Integrable X μ := by
  apply (integrable_const U).mono' hX.aestronglyMeasurable
  filter_upwards [hbounded] with ω hω
  simpa only [Real.norm_eq_abs, abs_of_nonneg hω.1] using hω.2

/-- A nonnegative bounded trait has zero expectation exactly when it is zero
almost surely. This is the zero-expectation claim used in the manuscript. -/
theorem expectation_eq_zero_iff_ae_zero [IsProbabilityMeasure μ]
    (hX : Measurable X)
    (hbounded : ∀ᵐ ω ∂μ, 0 ≤ X ω ∧ X ω ≤ U) :
    (∫ ω, X ω ∂μ) = 0 ↔ ∀ᵐ ω ∂μ, X ω = 0 := by
  have hnonneg : 0 ≤ᵐ[μ] X := hbounded.mono fun _ hω => hω.1
  exact integral_eq_zero_iff_of_nonneg_ae hnonneg
    (integrable_of_ae_bounded hX hbounded)

/-- Boundary innovation: if an offspring trait lies almost surely in `[0,U]`
and has positive probability of being positive, its expectation is positive.
For a parent at zero, this is also a strictly positive mean increment. -/
theorem boundary_innovation_positive_expectation [IsProbabilityMeasure μ]
    (hX : Measurable X)
    (hbounded : ∀ᵐ ω ∂μ, 0 ≤ X ω ∧ X ω ≤ U)
    (hinnovation : 0 < μ {ω | 0 < X ω}) :
    0 < ∫ ω, X ω ∂μ := by
  have hnonneg : 0 ≤ᵐ[μ] X := hbounded.mono fun _ hω => hω.1
  apply (integral_pos_iff_support_of_nonneg_ae hnonneg
    (integrable_of_ae_bounded hX hbounded)).2
  apply lt_of_lt_of_le hinnovation
  apply measure_mono
  intro ω hω
  exact ne_of_gt hω

/-- No bounded mutation rule can simultaneously preserve the mean at a
zero-valued parent and allow positive-probability positive innovation. -/
theorem no_mean_unbiased_boundary_innovation [IsProbabilityMeasure μ]
    (hX : Measurable X)
    (hbounded : ∀ᵐ ω ∂μ, 0 ≤ X ω ∧ X ω ≤ U) :
    ¬ ((∫ ω, X ω ∂μ) = 0 ∧ 0 < μ {ω | 0 < X ω}) := by
  intro h
  exact (ne_of_gt (boundary_innovation_positive_expectation hX hbounded h.2)) h.1

end BoundaryTransmission
