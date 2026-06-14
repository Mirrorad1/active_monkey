"""
ecology.evolvability — Evolvability Preflight framework.

Public API
----------
TraitAxis, THERMOSENSE_AXIS, make_axis  — trait/organ descriptor
PreflightConfig, ControllerAxis,        — config + loaders
    load_config, from_yaml, from_json
BenefitVerdict, GradientVerdict,        — verdict enums
    InvasionVerdict, CrossPartialVerdict,
    GuardStatus, AggregateVerdict
PreflightResult, run_preflight          — orchestration
"""

from .trait_axis import TraitAxis, THERMOSENSE_AXIS, make_axis
from .config import PreflightConfig, ControllerAxis, load_config, from_yaml, from_json
from .verdicts import (
    BenefitVerdict,
    GradientVerdict,
    InvasionVerdict,
    CrossPartialVerdict,
    GuardStatus,
    AggregateVerdict,
)
from .runner import PreflightResult, run_preflight

__all__ = [
    # trait axis
    "TraitAxis",
    "THERMOSENSE_AXIS",
    "make_axis",
    # config
    "PreflightConfig",
    "ControllerAxis",
    "load_config",
    "from_yaml",
    "from_json",
    # verdicts
    "BenefitVerdict",
    "GradientVerdict",
    "InvasionVerdict",
    "CrossPartialVerdict",
    "GuardStatus",
    "AggregateVerdict",
    # runner
    "PreflightResult",
    "run_preflight",
]
