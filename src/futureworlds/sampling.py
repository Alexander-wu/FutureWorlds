"""Original SFT sample schedules, independent of DDP rank count."""

import random
from functools import lru_cache
import numpy as np


class Schedule:
    def __init__(self, rows, dataset, seed):
        self.rows = rows
        self.dataset = dataset
        self.seed = seed
        if dataset == "robocasa":
            self.groups = {}
            for r in rows:
                self.groups.setdefault(r["task"], {}).setdefault(
                    r["episode_id"], []
                ).append(r)
            self.tasks = sorted(self.groups)

    @lru_cache(maxsize=4)
    def order(self, epoch):
        if self.dataset == "bridge":
            order = list(range(len(self.rows)))
            random.Random(self.seed + epoch).shuffle(order)
            return order
        return np.random.default_rng(self.seed + epoch).permutation(len(self.rows))

    def pick(self, index):
        if self.dataset == "robocasa":
            task = self.tasks[index % len(self.tasks)]
            rng = np.random.default_rng(self.seed + index * 9973)
            episodes = sorted(self.groups[task])
            eid = episodes[int(rng.integers(len(episodes)))]
            windows = self.groups[task][eid]
            row = windows[int(rng.integers(len(windows)))]
            target = int(
                np.random.default_rng(self.seed + index * 9973 + 23).integers(
                    2, row["frames"]
                )
            )
        else:
            epoch, j = divmod(index, len(self.rows))
            row = self.rows[int(self.order(epoch)[j])]
            if self.dataset == "bridge":
                target = random.Random(self.seed + 2000003 + index).randrange(
                    1, min(32, row["frames"] - 1) + 1
                )
            else:
                target = int(
                    np.random.default_rng(self.seed + index * 9973).integers(
                        2, row["frames"]
                    )
                )
        return row, target
