import BoundaryTransmission.Price
import BoundaryTransmission.Boundary
import BoundaryTransmission.Gaussian
import BoundaryTransmission.Examples
import Lean.Util.CollectAxioms

/-!
Reject additional axioms in every theorem in this package, including `sorryAx`.
The allowlist contains only Lean's conventional classical foundations.
This command inspects the kernel-checked declarations, not their source spelling.
-/

open Lean Elab Command in
run_cmd do
  let allowed := [``propext, ``Classical.choice, ``Quot.sound]
  let env ← getEnv
  let mut count := 0
  for (name, info) in env.constants.toList do
    if (`BoundaryTransmission).isPrefixOf name then
      match info with
      | .axiomInfo _ => throwError "Local axiom is forbidden: {name}"
      | .thmInfo _ =>
        let dependencies ← collectAxioms name
        for dependency in dependencies do
          unless allowed.contains dependency do
            throwError "Unexpected axiom {dependency} in {name}"
        count := count + 1
      | _ => pure ()
  if count == 0 then
    throwError "No BoundaryTransmission theorems found; audit cannot pass vacuously."
  logInfo m!"Axiom audit passed for {count} theorem declarations (including generated helpers); only propext, Classical.choice, Quot.sound allowed."
