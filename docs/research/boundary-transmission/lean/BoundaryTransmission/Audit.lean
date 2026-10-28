import BoundaryTransmission.Price
import BoundaryTransmission.Boundary
import BoundaryTransmission.Gaussian
import BoundaryTransmission.Examples
import Lean.Util.CollectAxioms

/-!
Reject additional axioms in every safe declaration in this package, including `sorryAx`.
The allowlist contains only Lean's conventional classical foundations.
This command inspects the kernel-checked declarations, not their source spelling.
Ownership follows the defining module, not the declaration's namespace.
Unsafe compiler specializations cannot enter safe proofs and are counted separately.
-/

open Lean Elab Command in
run_cmd do
  let allowed := [``propext, ``Classical.choice, ``Quot.sound]
  let env ← getEnv
  let mut count := 0
  let mut theoremCount := 0
  let mut unsafeCount := 0
  for (name, info) in env.constants.toList do
    let owner := match env.getModuleIdxFor? name with
      | some idx => env.header.moduleNames[idx.toNat]!
      | none => env.mainModule
    if (`BoundaryTransmission).isPrefixOf owner then
      if info.isUnsafe then
        unsafeCount := unsafeCount + 1
        continue
      match info with
      | .axiomInfo _ => throwError "Local axiom is forbidden: {name}"
      | _ => pure ()
      let dependencies ← collectAxioms name
      for dependency in dependencies do
        unless allowed.contains dependency do
          throwError "Unexpected axiom {dependency} in {name}"
      count := count + 1
      if let .thmInfo _ := info then
        theoremCount := theoremCount + 1
  if theoremCount == 0 then
    throwError "No BoundaryTransmission theorems found; audit cannot pass vacuously."
  logInfo m!"Axiom audit passed for {count} safe package declarations, including {theoremCount} theorem declarations and generated helpers; {unsafeCount} unsafe runtime declarations excluded; only propext, Classical.choice, Quot.sound allowed."
