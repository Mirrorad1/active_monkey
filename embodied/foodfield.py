"""embodied.foodfield — a shared, depletable resource field mediating competition."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FoodFieldConfig:
    extent: float = 6.0
    cells: int = 24
    capacity: float = 1.0
    regen: float = 0.02
    n_sources: int = 5


class FoodField:
    def __init__(self, cfg: FoodFieldConfig, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        n = cfg.cells
        xs = np.linspace(-cfg.extent, cfg.extent, n)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        field = np.zeros((n, n))
        centers = rng.uniform(-cfg.extent * 0.7, cfg.extent * 0.7, size=(cfg.n_sources, 2))
        for cx, cy in centers:
            field += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * (cfg.extent * 0.25) ** 2))
        self._cap = np.clip(field / field.max() * cfg.capacity, 1e-6, cfg.capacity)  # per-cell local cap
        self.grid = self._cap.copy()
        self._xs = xs

    def _ij(self, x, y):
        n = self.cfg.cells
        i = int(np.clip(round((x + self.cfg.extent) / (2 * self.cfg.extent) * (n - 1)), 0, n - 1))
        j = int(np.clip(round((y + self.cfg.extent) / (2 * self.cfg.extent) * (n - 1)), 0, n - 1))
        return i, j

    def resource_at(self, x, y) -> float:
        i, j = self._ij(x, y)
        return float(self.grid[i, j])

    def nearest_food_xy(self, x, y, thresh=0.05):
        n = self.cfg.cells
        mask = self.grid > thresh
        if not mask.any():
            i, j = np.unravel_index(int(np.argmax(self.grid)), self.grid.shape)
        else:
            X, Y = np.meshgrid(self._xs, self._xs, indexing="ij")
            d2 = (X - x) ** 2 + (Y - y) ** 2
            d2[~mask] = np.inf
            i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        return float(self._xs[i]), float(self._xs[j])

    def consume(self, path_xy, deficit) -> float:
        if deficit <= 0 or len(path_xy) < 1:
            return 0.0
        taken = 0.0
        for (x, y) in path_xy:
            i, j = self._ij(x, y)
            avail = float(self.grid[i, j])
            take = min(avail, max(0.0, deficit - taken))
            self.grid[i, j] = avail - take
            taken += take
            if taken >= deficit:
                break
        return taken

    def step_regen(self):
        self.grid = np.minimum(self._cap, self.grid + self.cfg.regen)

    def total(self) -> float:
        return float(self.grid.sum())
