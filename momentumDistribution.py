"""Post-processing helpers for final momentum-distribution profiles."""

import csv
from pathlib import Path
from typing import Optional, Tuple, Union

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
        self.n0 = nk_values[0]
        self.n0err = nkerr_values[0]

    def total_atom_number(self, k_cutoff: Optional[float] = None) -> Tuple[float, float]:
        """Return the integrated atom number and summed ``nk`` uncertainty.

        The atom number is obtained by integrating ``4*pi*k^2*nk`` over the
        finite, positive-k data.  When ``k_cutoff`` lies within the available
        k range, the final trapezoid is clipped at that value; otherwise the
        complete range is integrated.  The returned uncertainty is the sum of
        the finite ``nkerr`` values at k points included in the integration.
        """
        return self._integrate(k_power=2, prefactor=4.0 * np.pi, k_cutoff=k_cutoff)

    def total_energy(self, k_cutoff: Optional[float] = None) -> Tuple[float, float]:
        """Return the integrated energy and summed ``nk`` uncertainty.

        The energy is obtained by integrating ``2*pi*k^4*nk`` over the finite,
        positive-k data.  ``k_cutoff`` behaves as it does for
        :meth:`total_atom_number`.
        """
        return self._integrate(k_power=4, prefactor=2.0 * np.pi, k_cutoff=k_cutoff)

    def _integrate(
        self, k_power: int, prefactor: float, k_cutoff: Optional[float]
    ) -> Tuple[float, float]:
        """Integrate a k-weighted profile and sum included ``nk`` errors."""
        valid = np.isfinite(self.k) & np.isfinite(self.nk) & (self.k > 0)
        k = self.k[valid]
        nk = self.nk[valid]
        nkerr = self.nkerr[valid]
        if k.size < 2:
            raise ValueError("At least two finite, positive-k points are required for integration.")

        order = np.argsort(k)
        k = k[order]
        nk = nk[order]
        nkerr = nkerr[order]

        if k_cutoff is not None and k[-1] > k_cutoff:
            included = k <= k_cutoff
            k = k[included]
            nk = nk[included]
            nkerr = nkerr[included]

            # Include an interpolated endpoint so the integral terminates at
            # the requested cutoff, even when it falls between k samples.
            if k.size == 0 or k[-1] < k_cutoff:
                original_k = self.k[valid][order]
                original_nk = self.nk[valid][order]
                k = np.append(k, k_cutoff)
                nk = np.append(nk, np.interp(k_cutoff, original_k, original_nk))

        if k.size < 2:
            raise ValueError("At least two finite, positive-k points are required for integration.")

        integrand = prefactor * np.power(k, k_power) * nk
        error = np.sum(nkerr[np.isfinite(nkerr)])
        return float(np.trapz(integrand, k)), float(error)

    def ellsq(self, n0_bar=1500, R=21., L=42., zeta=1.9) -> Tuple[float, float]:
        """Return the square of the coherence length and its uncertainty using Gevorg's procedure."""
        V = 2 * np.pi * R**2 * L # um^3
        ell0 = V**(1/3) / (zeta**(2/3)-1)**0.5 # um

        ellprime = (self.n0 / (n0_bar*zeta) * V)**(1/3) # um
        ellprime_err = 1/3 * (self.n0err / self.n0) * ellprime

        ellsq = ellprime**2 / (1 - ellprime**2 / ell0**2)
        ellsq_err = ellprime_err * (2*ellprime / (1-ellprime**2/ell0**2) + 2*ellprime**3 / (ell0**2 * (1-ellprime**2/ell0**2)**2))

        return ellsq, ellsq_err