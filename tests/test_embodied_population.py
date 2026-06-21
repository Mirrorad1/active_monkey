import numpy as np, pytest
from pathlib import Path

def test_foodfield_deplete_regen_nearest():
    from embodied.foodfield import FoodField, FoodFieldConfig
    f = FoodField(FoodFieldConfig(extent=6.0, cells=24, capacity=1.0, regen=0.1, n_sources=5), seed=0)
    t0 = f.total()
    # consume along a path through the arena
    got = f.consume([(-3.0, 0.0), (3.0, 0.0)], deficit=10.0)
    assert got > 0.0 and got <= 10.0
    assert f.total() < t0                       # depletion happened
    f.step_regen()
    assert f.total() > (t0 - got)               # regen restored some
    # nearest_food points at a real high-resource cell
    fx, fy = f.nearest_food_xy(0.0, 0.0)
    assert -6.0 <= fx <= 6.0 and -6.0 <= fy <= 6.0
    assert f.resource_at(fx, fy) > 0.0
