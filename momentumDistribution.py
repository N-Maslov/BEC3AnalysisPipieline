"""Post-processing helpers for final momentum-distribution profiles."""

import csv
from pathlib import Path
from typing import Union

import numpy as np


class MomentumDistribution:
    """A final momentum distribution loaded from a pipeline CSV output."""

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        k_values = []
        nk_values = []
        nkerr_values = []

        try:
            with self.filepath.open(newline="") as profile_file:
                reader = csv.DictReader(profile_file)
                required_columns = {"k", "nk", "stderr"}
                if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                    raise ValueError(
                        "Momentum-distribution CSV must contain k, nk, and stderr columns."
                    )
                for row in reader:
                    k_values.append(float(row["k"]))
                    nk_values.append(float(row["nk"]))
                    nkerr_values.append(float(row["stderr"]))
        except OSError as exc:
            raise ValueError(f"Could not read momentum-distribution file '{self.filepath}': {exc}") from exc

        self.k = np.asarray(k_values, dtype=float)
        self.nk = np.asarray(nk_values, dtype=float)
        self.nkerr = np.asarray(nkerr_values, dtype=float)

    def total_atom_number(self) -> float:
        """Numerically integrate 4*pi*k^2*nk over the finite, positive k range."""
        valid = np.isfinite(self.k) & np.isfinite(self.nk) & (self.k > 0)
        k = self.k[valid]
        nk = self.nk[valid]
        if k.size < 2:
            raise ValueError("At least two finite, positive-k points are required for integration.")

        order = np.argsort(k)
        integrand = 4.0 * np.pi * np.square(k[order]) * nk[order]
        return float(np.trapz(integrand, k[order]))
