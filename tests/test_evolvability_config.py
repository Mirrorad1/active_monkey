"""
tests/test_evolvability_config.py — unit tests for evolvability config + trait_axis.
"""
import json
import pytest

from ecology.evolvability.config import PreflightConfig, load_config
from ecology.evolvability.trait_axis import (
    TraitAxis,
    THERMOSENSE_AXIS,
    make_axis,
)
from ecology.genotype import founder


# ---------------------------------------------------------------------------
# PreflightConfig: basic construction
# ---------------------------------------------------------------------------

class TestPreflightConfigDefaults:
    def test_builds_with_defaults(self):
        cfg = PreflightConfig(slug="test")
        assert cfg.slug == "test"

    def test_trait_defaults_to_thermosense(self):
        cfg = PreflightConfig(slug="test")
        assert cfg.trait.name == "thermosense"

    def test_seeds_is_tuple(self):
        cfg = PreflightConfig(slug="test")
        assert isinstance(cfg.seeds, tuple)

    def test_monomorphic_grid_is_tuple(self):
        cfg = PreflightConfig(slug="test")
        assert isinstance(cfg.monomorphic_grid, tuple)


# ---------------------------------------------------------------------------
# Round-trip: to_dict / from_dict
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def _make_cfg(self):
        return PreflightConfig(
            slug="roundtrip-test",
            description="a round-trip fixture",
            seeds=(1, 2, 3, 4),
            monomorphic_grid=(0.0, 0.10, 0.30, 0.60),
            horizon=2000,
        )

    def test_slug_survives(self):
        cfg = self._make_cfg()
        cfg2 = PreflightConfig.from_dict(cfg.to_dict())
        assert cfg2.slug == cfg.slug

    def test_seeds_tuple_survives(self):
        cfg = self._make_cfg()
        cfg2 = PreflightConfig.from_dict(cfg.to_dict())
        assert cfg2.seeds == cfg.seeds
        assert isinstance(cfg2.seeds, tuple)

    def test_trait_name_survives(self):
        cfg = self._make_cfg()
        cfg2 = PreflightConfig.from_dict(cfg.to_dict())
        assert cfg2.trait.name == cfg.trait.name

    def test_monomorphic_grid_survives(self):
        cfg = self._make_cfg()
        cfg2 = PreflightConfig.from_dict(cfg.to_dict())
        assert cfg2.monomorphic_grid == cfg.monomorphic_grid
        assert isinstance(cfg2.monomorphic_grid, tuple)


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------

class TestConfigHash:
    def test_deterministic(self):
        cfg = PreflightConfig(slug="hash-test")
        assert cfg.config_hash() == cfg.config_hash()

    def test_changes_with_slug(self):
        cfg_a = PreflightConfig(slug="alpha")
        cfg_b = PreflightConfig(slug="beta")
        assert cfg_a.config_hash() != cfg_b.config_hash()

    def test_changes_with_horizon(self):
        cfg_a = PreflightConfig(slug="x", horizon=1000)
        cfg_b = PreflightConfig(slug="x", horizon=2000)
        assert cfg_a.config_hash() != cfg_b.config_hash()


# ---------------------------------------------------------------------------
# from_dict: trait as string or dict
# ---------------------------------------------------------------------------

class TestFromDictTrait:
    def test_trait_as_string(self):
        cfg = PreflightConfig.from_dict({"slug": "s", "trait": "thermosense"})
        assert cfg.trait.name == "thermosense"

    def test_trait_as_dict(self):
        axis_d = THERMOSENSE_AXIS.to_dict()
        cfg = PreflightConfig.from_dict({"slug": "s", "trait": axis_d})
        assert cfg.trait.name == "thermosense"

    def test_unknown_keys_ignored(self):
        cfg = PreflightConfig.from_dict({"slug": "s", "__unknown__": 42})
        assert cfg.slug == "s"


# ---------------------------------------------------------------------------
# effective_thresholds
# ---------------------------------------------------------------------------

class TestEffectiveThresholds:
    def test_derived_from_seeds_n8(self):
        cfg = PreflightConfig(slug="t", seeds=tuple(range(8)))
        win, lose = cfg.effective_thresholds()
        assert win == 7
        assert lose == 3

    def test_explicit_overrides(self):
        cfg = PreflightConfig(slug="t", win_threshold=6, lose_threshold=2)
        win, lose = cfg.effective_thresholds()
        assert win == 6
        assert lose == 2

    def test_explicit_partial_still_uses_derive_if_either_none(self):
        # only win set but lose is None => both derived
        cfg = PreflightConfig(slug="t", seeds=tuple(range(8)), win_threshold=6, lose_threshold=None)
        win, lose = cfg.effective_thresholds()
        # win=6 but lose is None, so from_dict logic: effective_thresholds derives both
        assert lose == 3   # derived


# ---------------------------------------------------------------------------
# JSON round-trip via load_config
# ---------------------------------------------------------------------------

class TestJSONRoundTrip:
    def test_load_config_json(self, tmp_path):
        cfg = PreflightConfig(slug="json-test", horizon=999)
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
        cfg2 = load_config(p)
        assert cfg2.slug == cfg.slug
        assert cfg2.horizon == cfg.horizon
        assert cfg2.trait.name == cfg.trait.name

    def test_load_config_unsupported_suffix(self, tmp_path):
        p = tmp_path / "cfg.toml"
        p.write_text("slug = 'x'", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported"):
            load_config(p)


# ---------------------------------------------------------------------------
# YAML round-trip (skipped if PyYAML absent)
# ---------------------------------------------------------------------------

class TestYAMLRoundTrip:
    def test_load_config_yaml(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        cfg = PreflightConfig(slug="yaml-test", horizon=777)
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg.to_dict()), encoding="utf-8")
        cfg2 = load_config(p)
        assert cfg2.slug == cfg.slug
        assert cfg2.horizon == cfg.horizon


# ---------------------------------------------------------------------------
# TraitAxis: get / set / clamp / cost
# ---------------------------------------------------------------------------

class TestTraitAxis:
    def test_get_returns_zero_for_founder(self):
        g = founder()
        ax = THERMOSENSE_AXIS
        assert ax.get(g) == 0.0

    def test_set_returns_updated_genotype(self):
        g = founder()
        ax = THERMOSENSE_AXIS
        g2 = ax.set(g, 0.15)
        assert ax.get(g2) == pytest.approx(0.15)

    def test_clamp_upper(self):
        ax = THERMOSENSE_AXIS
        assert ax.clamp(99) == pytest.approx(1.0)   # TRAIT_BOUNDS upper for thermosense_intensity

    def test_clamp_lower(self):
        ax = THERMOSENSE_AXIS
        assert ax.clamp(-1) == pytest.approx(0.0)

    def test_cost_zero_below_threshold(self):
        ax = THERMOSENSE_AXIS
        assert ax.cost(0.0) == 0.0
        assert ax.cost(0.04) == 0.0   # below active_threshold=0.05

    def test_cost_at_threshold_boundary(self):
        ax = THERMOSENSE_AXIS
        # exactly at threshold => 0.0 (not above)
        assert ax.cost(0.05) == 0.0

    def test_cost_above_threshold(self):
        ax = THERMOSENSE_AXIS
        # h=0.10 => cost_floor(0) + cost_inefficiency(0.20) * 0.10 = 0.02
        assert ax.cost(0.10) == pytest.approx(0.02)

    def test_cost_monotone(self):
        ax = THERMOSENSE_AXIS
        assert ax.cost(0.60) > ax.cost(0.10)


# ---------------------------------------------------------------------------
# make_axis
# ---------------------------------------------------------------------------

class TestMakeAxis:
    def test_string_lookup(self):
        ax = make_axis("thermosense")
        assert ax.name == "thermosense"

    def test_unknown_string_raises_value_error(self):
        with pytest.raises(ValueError, match="nonexistent"):
            make_axis("nonexistent")

    def test_dict_constructs(self):
        d = THERMOSENSE_AXIS.to_dict()
        ax = make_axis(d)
        assert ax.name == "thermosense"
        assert isinstance(ax, TraitAxis)
