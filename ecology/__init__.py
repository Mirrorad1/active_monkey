"""ecology — population-ecology simulation substrate for active-loop Exp 194+.

Provides a deterministic, seed-controlled gridworld ecology for studying how
environment-driven selection (energy constraints + finite resources) shapes
population dynamics without any global fitness ranking.

Key modules:
  genotype   — Genotype dataclass, mutation, validation, clamping
  world      — GridWorld with regenerating resources, local sensing
  creature   — Phenotype, Policy protocol, HomeostaticPolicy, Creature
  engine     — EcologyConfig, Ecology (step/run loop)
  scenarios  — Pre-built scenario configs (balanced / scarce / overabundant)
  recording  — Output writers (JSONL events, CSV traits, lineage JSON, verdict)
  run        — run_scenario / determinism_check helpers
"""
from __future__ import annotations
