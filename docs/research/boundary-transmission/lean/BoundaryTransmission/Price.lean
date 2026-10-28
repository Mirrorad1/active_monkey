import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic

/-!
# Finite-census selection and transmission

The populations are nonempty finite types. An arbitrary total ancestry map sends
every after-census individual to its before-census ancestor. The map may include
self-survival and offspring of a parent that subsequently died. No claim about
causal effects, stochastic sampling, or the correctness of a simulator is assumed.
-/

open scoped BigOperators

namespace BoundaryTransmission

noncomputable section

/-- Census mean. Requiring `Nonempty` rules out an extinction census. -/
def mean {ι : Type*} [Fintype ι] [Nonempty ι] (f : ι → ℝ) : ℝ :=
  (∑ i, f i) / Fintype.card ι

variable {P Q : Type*} [Fintype P] [Nonempty P] [Fintype Q] [Nonempty Q]

theorem mean_sub (f g : Q → ℝ) :
    mean (fun j => f j - g j) = mean f - mean g := by
  simp [mean, Finset.sum_sub_distrib, sub_div]

/-- Realized differential persistence/reproduction, not a causal effect. -/
def selection (h : P → ℝ) (ancestor : Q → P) : ℝ :=
  mean (fun j => h (ancestor j)) - mean h

/-- Mean child-minus-ancestor change over the complete after-census. -/
def transmission (h : P → ℝ) (k : Q → ℝ) (ancestor : Q → P) : ℝ :=
  mean (fun j => k j - h (ancestor j))

/-- Paper Eq. (5), one transition, derived from finite sums. -/
theorem price_identity (h : P → ℝ) (k : Q → ℝ) (ancestor : Q → P) :
    mean k - mean h = selection h ancestor + transmission h k ancestor := by
  rw [transmission, mean_sub, selection]
  ring

section Counts

variable [DecidableEq P]

/-- The number of after-census descendants, expressed as a real-valued count. -/
def descendantWeight (ancestor : Q → P) (i : P) : ℝ :=
  ∑ j, if ancestor j = i then 1 else 0

omit [Fintype P] [Nonempty P] [Nonempty Q] in
theorem descendantWeight_eq_card (ancestor : Q → P) (i : P) :
    descendantWeight ancestor i =
      ((Finset.univ.filter (fun j => ancestor j = i)).card : ℝ) := by
  classical
  simp [descendantWeight]

omit [Nonempty P] [Nonempty Q] in
theorem sum_weighted_ancestry (h : P → ℝ) (ancestor : Q → P) :
    (∑ i, descendantWeight ancestor i * h i) = ∑ j, h (ancestor j) := by
  classical
  simp only [descendantWeight, Finset.sum_mul]
  rw [Finset.sum_comm]
  congr 1
  ext j
  simp [ite_mul]

omit [Nonempty P] [Nonempty Q] in
theorem sum_descendantWeight (ancestor : Q → P) :
    (∑ i, descendantWeight ancestor i) = (Fintype.card Q : ℝ) := by
  simpa using sum_weighted_ancestry (fun _ : P => (1 : ℝ)) ancestor

/-- Population covariance, with denominator N rather than N-1. -/
def covariance (f g : P → ℝ) : ℝ :=
  mean (fun i => f i * g i) - mean f * mean g

/-- The usual Price covariance form is exactly the ancestral-mean form. -/
theorem selection_eq_covariance (h : P → ℝ) (ancestor : Q → P) :
    selection h ancestor =
      covariance (descendantWeight ancestor) h / mean (descendantWeight ancestor) := by
  have hp : (Fintype.card P : ℝ) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  have hq : (Fintype.card Q : ℝ) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  simp only [selection, covariance, mean, sum_weighted_ancestry, sum_descendantWeight]
  field_simp
  ring

end Counts

omit [Fintype P] [Nonempty P] in
/-- Unchanged survivors contribute zero; births retain the full census denominator. -/
theorem transmission_only_newborns (h : P → ℝ) (k : Q → ℝ)
    (ancestor : Q → P) (newborn : Q → Prop) [DecidablePred newborn]
    (survivors_unchanged : ∀ j, ¬ newborn j → k j = h (ancestor j)) :
    transmission h k ancestor =
      (∑ j ∈ Finset.univ.filter newborn, (k j - h (ancestor j))) /
        (Fintype.card Q : ℝ) := by
  unfold transmission mean
  congr 1
  rw [Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : newborn j
  · simp [hj]
  · simp [hj, survivors_unchanged j hj]

/-- The cumulative identity telescopes, allowing population size to change. -/
theorem cumulative_price (m S T : ℕ → ℝ) (H : ℕ)
    (step : ∀ t < H, m (t + 1) - m t = S t + T t) :
    m H - m 0 = (∑ t ∈ Finset.range H, S t) + ∑ t ∈ Finset.range H, T t := by
  have telescopes : ∀ n, (∑ t ∈ Finset.range n, (m (t + 1) - m t)) = m n - m 0 := by
    intro n
    induction n with
    | zero => simp
    | succ n ih => rw [Finset.sum_range_succ, ih]; ring
  rw [← telescopes H, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun t ht => step t (Finset.mem_range.mp ht))

/-- A newborn-only endpoint still contains parent weighting and transmission. -/
theorem newborn_endpoint_identity (h : P → ℝ) (k : Q → ℝ) (parent : Q → P) :
    mean k - mean h =
      (mean (fun j => h (parent j)) - mean h) +
        mean (fun j => k j - h (parent j)) :=
  price_identity h k parent

/-- Increased mean with a nonpositive differential requires positive transmission. -/
theorem positive_growth_requires_transmission (h : P → ℝ) (k : Q → ℝ)
    (ancestor : Q → P) (growth : mean h < mean k)
    (downward : selection h ancestor ≤ 0) : 0 < transmission h k ancestor := by
  have identity := price_identity h k ancestor
  linarith

/-- Treatment contrasts retain the difference in transmission contributions. -/
theorem contrast_identity (g₁ g₂ s₁ s₂ t₁ t₂ : ℝ)
    (h₁ : g₁ = s₁ + t₁) (h₂ : g₂ = s₂ + t₂) :
    g₁ - g₂ = (s₁ - s₂) + (t₁ - t₂) := by
  rw [h₁, h₂]
  ring

end

end BoundaryTransmission
