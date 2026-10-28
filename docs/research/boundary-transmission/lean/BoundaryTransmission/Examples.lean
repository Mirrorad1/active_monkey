import BoundaryTransmission.Price

/-!
An exact two-step witness of the accounting phenomenon. This is an abstract
lineage ledger, not a claim that these particular births occur in the simulator.
It also checks nonuniform descendant weighting on a small explicit population.
-/

namespace BoundaryTransmission

noncomputable section

def founders : Fin 2 → ℝ := fun _ => 0
def intermediate : Fin 2 → ℝ := ![0, 2 / 5]
def finalCensus : Fin 1 → ℝ := fun _ => 1 / 10
def firstAncestry : Fin 2 → Fin 2 := id
def secondAncestry : Fin 1 → Fin 2 := fun _ => 0

/-- All founders are zero, and all values in both later censuses stay in [0,1]. -/
theorem example_bounds :
    (∀ i, founders i = 0) ∧
    (∀ i, 0 ≤ intermediate i ∧ intermediate i ≤ 1) ∧
    (∀ i, 0 ≤ finalCensus i ∧ finalCensus i ≤ 1) := by
  norm_num [founders, intermediate, finalCensus, Fin.forall_fin_two]

/-- From a zero founder mean, total growth is positive while cumulative S is negative. -/
theorem growth_against_selection_example :
    mean finalCensus - mean founders = 1 / 10 ∧
    selection founders firstAncestry + selection intermediate secondAncestry = -(1 / 5) ∧
    transmission founders intermediate firstAncestry +
      transmission intermediate finalCensus secondAncestry = 3 / 10 := by
  norm_num [mean, selection, transmission, founders, intermediate, finalCensus,
    firstAncestry, secondAncestry, Fin.sum_univ_two]

end

end BoundaryTransmission
