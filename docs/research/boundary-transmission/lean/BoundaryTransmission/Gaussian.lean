import Mathlib.Probability.Distributions.Gaussian
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Tactic

/-!
# Exact clipped-Gaussian transmission drift

The density and expectation here are the actual standard Gaussian density and
measure in mathlib. No truncated-moment identity or clipping formula is assumed.
-/

open MeasureTheory Set
open scoped Interval

namespace BoundaryTransmission

noncomputable def standardDensity (x : ℝ) : ℝ :=
  ProbabilityTheory.gaussianPDFReal 0 1 x

noncomputable def standardCDF (x : ℝ) : ℝ :=
  ∫ z in Iic x, standardDensity z

noncomputable def clippedValue (L U h σ z : ℝ) : ℝ :=
  min U (max L (h + σ * z))

lemma standardDensity_continuous : Continuous standardDensity := by
  unfold standardDensity ProbabilityTheory.gaussianPDFReal
  fun_prop

lemma standardDensity_integrable : Integrable standardDensity :=
  ProbabilityTheory.integrable_gaussianPDFReal 0 1

lemma standardDensity_integral : (∫ z, standardDensity z) = 1 := by
  exact ProbabilityTheory.integral_gaussianPDFReal_eq_one 0 one_ne_zero

lemma standardDensity_hasDerivAt (x : ℝ) :
    HasDerivAt standardDensity (-x * standardDensity x) x := by
  have he := ((((hasDerivAt_id x).pow 2).neg.div_const 2).exp).const_mul
    (1 / Real.sqrt (2 * Real.pi))
  convert he using 1
  · funext y
    simp only [standardDensity, ProbabilityTheory.gaussianPDFReal, NNReal.coe_one,
      sub_zero, mul_one, id_eq, one_div]
  · simp only [standardDensity, ProbabilityTheory.gaussianPDFReal, NNReal.coe_one,
      sub_zero, mul_one, id_eq, one_div]
    ring

/-- The interior Gaussian first moment, proved by differentiating the density. -/
theorem standardGaussian_firstMoment (a b : ℝ) :
    (∫ z in a..b, z * standardDensity z) = standardDensity a - standardDensity b := by
  have hd : ∀ z ∈ uIcc a b,
      HasDerivAt (fun x => -standardDensity x) (z * standardDensity z) z := by
    intro z _
    convert (standardDensity_hasDerivAt z).neg using 1
    ring
  have hi : IntervalIntegrable (fun z => z * standardDensity z) volume a b :=
    (continuous_id.mul standardDensity_continuous).intervalIntegrable _ _
  have ht := intervalIntegral.integral_eq_sub_of_hasDerivAt hd hi
  linarith

lemma standardDensity_upperTail (b : ℝ) :
    (∫ z in Ioi b, standardDensity z) = 1 - standardCDF b := by
  have ht := intervalIntegral.integral_Iic_add_Ioi
    (standardDensity_integrable.integrableOn (s := Iic b))
    (standardDensity_integrable.integrableOn (s := Ioi b))
  rw [standardDensity_integral] at ht
  unfold standardCDF
  linarith

lemma clippedValue_lower {L U h σ z : ℝ} (hLU : L ≤ U) (hσ : 0 < σ)
    (hz : z ≤ (L - h) / σ) : clippedValue L U h σ z = L := by
  have hz' := (le_div_iff₀ hσ).mp hz
  have hl : h + σ * z ≤ L := by nlinarith
  simp [clippedValue, max_eq_left hl, min_eq_right hLU]

lemma clippedValue_upper {L U h σ z : ℝ} (hLU : L ≤ U) (hσ : 0 < σ)
    (hz : (U - h) / σ ≤ z) : clippedValue L U h σ z = U := by
  have hz' := (div_le_iff₀ hσ).mp hz
  have hu : U ≤ h + σ * z := by nlinarith
  simp [clippedValue, max_eq_right (hLU.trans hu), min_eq_left hu]

lemma clippedValue_interior {L U h σ z : ℝ} (hσ : 0 < σ)
    (hl : (L - h) / σ ≤ z) (hu : z ≤ (U - h) / σ) :
    clippedValue L U h σ z = h + σ * z := by
  have hl' := (div_le_iff₀ hσ).mp hl
  have hu' := (le_div_iff₀ hσ).mp hu
  have hl'' : L ≤ h + σ * z := by nlinarith
  have hu'' : h + σ * z ≤ U := by nlinarith
  simp [clippedValue, max_eq_right hl'', min_eq_right hu'']

lemma clippedDisplacement_integrable (L U h σ : ℝ) (hLU : L ≤ U) :
    Integrable (fun z => (clippedValue L U h σ z - h) * standardDensity z) := by
  apply standardDensity_integrable.bdd_mul
  · have hc : Continuous (fun z => clippedValue L U h σ z - h) := by
      unfold clippedValue
      fun_prop
    exact hc.aestronglyMeasurable
  · refine ⟨|L| + |U| + |h|, ?_⟩
    intro z
    have hl : L ≤ clippedValue L U h σ z :=
      le_min hLU (le_max_left _ _)
    have hu : clippedValue L U h σ z ≤ U := min_le_left _ _
    rw [Real.norm_eq_abs]
    apply abs_le.mpr
    constructor
    · have := neg_abs_le L
      have := le_abs_self h
      have := abs_nonneg U
      linarith
    · have := le_abs_self U
      have := neg_abs_le h
      have := abs_nonneg L
      linarith

/-- The Gaussian expectation below is integrable, independently of its value. -/
lemma clippedDisplacement_gaussianIntegrable (L U h σ : ℝ) (hLU : L ≤ U) :
    Integrable (fun z => clippedValue L U h σ z - h)
      (ProbabilityTheory.gaussianReal 0 1) := by
  rw [ProbabilityTheory.gaussianReal_of_var_ne_zero 0 one_ne_zero,
    integrable_withDensity_iff_integrable_smul'
      (ProbabilityTheory.measurable_gaussianPDF 0 1)
      (ae_of_all _ fun z => ProbabilityTheory.gaussianPDF_lt_top)]
  simpa only [ProbabilityTheory.toReal_gaussianPDF, smul_eq_mul,
    standardDensity, mul_comm] using clippedDisplacement_integrable L U h σ hLU

/-- Equation (2), as a Lebesgue integral against the actual normal density. -/
theorem clippedGaussian_drift_density (L U h σ : ℝ) (hLU : L ≤ U) (hσ : 0 < σ) :
    (∫ z, (clippedValue L U h σ z - h) * standardDensity z) =
      σ * (standardDensity ((L - h) / σ) - standardDensity ((U - h) / σ)) +
      (L - h) * standardCDF ((L - h) / σ) +
      (U - h) * (1 - standardCDF ((U - h) / σ)) := by
  let a := (L - h) / σ
  let b := (U - h) / σ
  let q := fun z => (clippedValue L U h σ z - h) * standardDensity z
  have hab : a ≤ b := div_le_div_of_nonneg_right (by linarith) hσ.le
  have hq : Integrable q := clippedDisplacement_integrable L U h σ hLU
  have hlo : (∫ z in Iic a, q z) = (L - h) * standardCDF a := by
    calc
      _ = ∫ z in Iic a, (L - h) * standardDensity z := by
        apply setIntegral_congr_fun measurableSet_Iic
        intro z hz
        dsimp [q]
        rw [clippedValue_lower hLU hσ hz]
      _ = _ := by rw [integral_const_mul]; rfl
  have hhi : (∫ z in Ioi b, q z) = (U - h) * (1 - standardCDF b) := by
    calc
      _ = ∫ z in Ioi b, (U - h) * standardDensity z := by
        apply setIntegral_congr_fun measurableSet_Ioi
        intro z hz
        dsimp [q]
        rw [clippedValue_upper hLU hσ (le_of_lt hz)]
      _ = _ := by rw [integral_const_mul, standardDensity_upperTail]
  have hmid : (∫ z in a..b, q z) = σ * (standardDensity a - standardDensity b) := by
    calc
      _ = ∫ z in a..b, σ * (z * standardDensity z) := by
        apply intervalIntegral.integral_congr
        intro z hz
        rw [uIcc_of_le hab] at hz
        dsimp [q]
        rw [clippedValue_interior hσ hz.1 hz.2]
        ring
      _ = _ := by rw [intervalIntegral.integral_const_mul, standardGaussian_firstMoment]
  have hsplit := intervalIntegral.integral_Iic_sub_Iic
    (hq.integrableOn (s := Iic a)) (hq.integrableOn (s := Iic b))
  have hall := intervalIntegral.integral_Iic_add_Ioi
    (hq.integrableOn (s := Iic b)) (hq.integrableOn (s := Ioi b))
  rw [hlo, hmid] at hsplit
  rw [hhi] at hall
  change (∫ z, q z) = _
  change (∫ z, q z) = σ * (standardDensity a - standardDensity b) +
    (L - h) * standardCDF a + (U - h) * (1 - standardCDF b)
  linarith

/-- Equation (2) for expectation under mathlib's standard Gaussian probability measure. -/
theorem clippedGaussian_drift (L U h σ : ℝ) (hLU : L ≤ U) (hσ : 0 < σ) :
    (∫ z, clippedValue L U h σ z - h ∂ProbabilityTheory.gaussianReal 0 1) =
      σ * (standardDensity ((L - h) / σ) - standardDensity ((U - h) / σ)) +
      (L - h) * standardCDF ((L - h) / σ) +
      (U - h) * (1 - standardCDF ((U - h) / σ)) := by
  rw [ProbabilityTheory.integral_gaussianReal_eq_integral_smul one_ne_zero]
  simp only [smul_eq_mul]
  simpa only [standardDensity, mul_comm] using
    clippedGaussian_drift_density L U h σ hLU hσ

end BoundaryTransmission
