"""Post-processing helpers for final momentum-distribution profiles."""

import csv
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

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
        """Return the integrated atom number and its propagated uncertainty.

        The atom number is obtained by integrating ``4*pi*k^2*nk`` over the
        finite, positive-k data.  When ``k_cutoff`` lies within the available
        k range, the final trapezoid is clipped at that value; otherwise the
        complete range is integrated.  The returned uncertainty is propagated
        through the trapezoidal integral, assuming independent ``nk`` errors.
        """
        return self._integrate(k_power=2, prefactor=4.0 * np.pi, k_cutoff=k_cutoff)

    def total_energy(self, k_cutoff: Optional[float] = None, per_particle: bool = True) -> Tuple[float, float]:
        """Return the integrated energy and its propagated uncertainty.

        The energy is obtained by integrating ``2*pi*k^4*nk`` over the finite,
        positive-k data.  ``k_cutoff`` behaves as it does for
        :meth:`total_atom_number`.
        """
        energy, energy_err = self._integrate(k_power=4, prefactor=2.0 * np.pi * 12.4497, k_cutoff=k_cutoff)
        if per_particle:
            N, Nerr = self.total_atom_number(k_cutoff=k_cutoff)
            energy /= N
            energy_err = np.sqrt((energy_err / N) ** 2 + (energy * Nerr / N**2) ** 2)
        return energy, energy_err

    def inner_product(
        self,
        other: "MomentumDistribution",
        weighting_function: Callable[[np.ndarray], Union[float, np.ndarray]],
        k_cutoff: float,
        subtraction: bool = False
    ) -> Tuple[float, float]:
        """Return the weighted overlap and its propagated uncertainty.

        This evaluates
        ``integral_0^k_cutoff w(k) * min(n1(k)/N1, n2(k)/N2) dk``, where
        ``N1`` and ``N2`` are each distribution's total atom number.  The
        profiles are linearly interpolated onto their combined sampled
        momenta, including the requested cutoff when it lies between samples.
        Integration begins at the later of the two profiles' first measured
        non-negative momentum values, so neither profile is extrapolated
        below its data.  Both profiles must extend through ``k_cutoff``.

        The uncertainty includes both ``nk`` and total-atom-number errors in
        ``nk / N``.  The error of the selected integrand is integrated with
        the trapezoidal weights and combined in quadrature.
        """
        if not isinstance(other, MomentumDistribution):
            raise TypeError("other must be a MomentumDistribution.")
        if not callable(weighting_function):
            raise TypeError("weighting_function must be callable.")
        if not np.isfinite(k_cutoff) or k_cutoff <= 0:
            raise ValueError("k_cutoff must be a positive finite value.")

        self_k, self_nk, self_nkerr = self._profile_for_overlap(k_cutoff)
        other_k, other_nk, other_nkerr = other._profile_for_overlap(k_cutoff)
        atom_number, atom_number_err = self.total_atom_number()
        other_atom_number, other_atom_number_err = other.total_atom_number()
        if atom_number == 0 or other_atom_number == 0:
            raise ValueError("Both distributions must have non-zero total atom number.")

        lower_bound = max(self_k[0], other_k[0])
        k = np.union1d(self_k, other_k)
        k = k[k >= lower_bound]
        self_nk = np.interp(k, self_k, self_nk)
        other_nk = np.interp(k, other_k, other_nk)
        normalized_self = self_nk / atom_number
        normalized_other = other_nk / other_atom_number
        normalized_self_err = np.hypot(
            np.interp(k, self_k, self_nkerr) / atom_number,
            self_nk * atom_number_err / atom_number**2,
        )
        normalized_other_err = np.hypot(
            np.interp(k, other_k, other_nkerr) / other_atom_number,
            other_nk * other_atom_number_err / other_atom_number**2,
        )
        weight = np.asarray(weighting_function(k), dtype=float)
        if weight.ndim == 0:
            weight = np.full_like(k, weight)
        elif weight.shape != k.shape:
            raise ValueError("weighting_function must return a scalar or one value per k.")
        if not np.all(np.isfinite(weight)):
            raise ValueError("weighting_function must return finite values.")

        if subtraction:
            integrand = weight * np.abs(normalized_self - normalized_other)
            integrand_err = np.abs(weight) * np.hypot(
                normalized_self_err, normalized_other_err
            )
        else:
            self_is_minimum = normalized_self <= normalized_other
            integrand = weight * np.where(
                self_is_minimum, normalized_self, normalized_other
            )
            integrand_err = np.abs(weight) * np.where(
                self_is_minimum, normalized_self_err, normalized_other_err
            )

        trapezoid_weights = np.empty_like(k)
        trapezoid_weights[0] = (k[1] - k[0]) / 2.0
        trapezoid_weights[-1] = (k[-1] - k[-2]) / 2.0
        trapezoid_weights[1:-1] = (k[2:] - k[:-2]) / 2.0
        error = np.sqrt(np.sum(np.square(trapezoid_weights * integrand_err)))
        return float(np.trapz(integrand, k)), float(error)

    def k_p(self, max_k: Optional[float] = None) -> float:
        """Return the sampled momentum where ``k**2 * nk`` is largest.

        When given, ``max_k`` restricts the search to sampled values at or
        below that momentum.
        """
        return self._peak_k(k_power=2, max_k=max_k)

    def k_epsilon(self, max_k: Optional[float] = None) -> float:
        """Return the sampled momentum where ``k**4 * nk`` is largest.

        When given, ``max_k`` restricts the search to sampled values at or
        below that momentum.
        """
        return self._peak_k(k_power=4, max_k=max_k)

    def _peak_k(self, k_power: int, max_k: Optional[float] = None) -> float:
        """Return the k value at the maximum finite k-weighted occupation."""
        valid = np.isfinite(self.k) & np.isfinite(self.nk)
        if max_k is not None:
            valid &= self.k <= max_k
        if not np.any(valid):
            raise ValueError("At least one finite k and nk value is required to find a peak in the requested range.")

        k = self.k[valid]
        nk = self.nk[valid]
        return float(k[np.argmax(np.power(k, k_power) * nk)])

    def _profile_for_overlap(self, k_cutoff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a finite profile clipped to an overlap integration range."""
        valid = np.isfinite(self.k) & np.isfinite(self.nk)
        k = self.k[valid]
        nk = self.nk[valid]
        nkerr = self.nkerr[valid]
        if k.size < 2:
            raise ValueError("At least two finite k and nk points are required for an inner product.")

        order = np.argsort(k)
        k = k[order]
        nk = nk[order]
        nkerr = nkerr[order]
        if np.any(np.diff(k) == 0):
            raise ValueError("Momentum values must be unique for an inner product.")
        non_negative = k >= 0
        k = k[non_negative]
        nk = nk[non_negative]
        nkerr = nkerr[non_negative]
        if k.size < 2 or k[0] > k_cutoff or k[-1] < k_cutoff:
            raise ValueError("Both distributions must contain at least two non-negative k values and extend through k_cutoff.")

        included = k <= k_cutoff
        clipped_k = k[included]
        clipped_nk = nk[included]
        clipped_nkerr = nkerr[included]
        if clipped_k[-1] < k_cutoff:
            upper_index = np.searchsorted(k, k_cutoff)
            lower_index = upper_index - 1
            fraction = (k_cutoff - k[lower_index]) / (k[upper_index] - k[lower_index])
            clipped_k = np.append(clipped_k, k_cutoff)
            clipped_nk = np.append(clipped_nk, np.interp(k_cutoff, k, nk))
            clipped_nkerr = np.append(
                clipped_nkerr,
                np.hypot(
                    (1.0 - fraction) * nkerr[lower_index],
                    fraction * nkerr[upper_index],
                ),
            )
        return clipped_k, clipped_nk, clipped_nkerr

    def _integrate(
        self, k_power: int, prefactor: float, k_cutoff: Optional[float]
    ) -> Tuple[float, float]:
        """Integrate a k-weighted profile and propagate ``nk`` uncertainties."""
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
                original_nkerr = self.nkerr[valid][order]
                upper_index = np.searchsorted(original_k, k_cutoff)
                lower_index = upper_index - 1
                fraction = (
                    (k_cutoff - original_k[lower_index])
                    / (original_k[upper_index] - original_k[lower_index])
                )
                k = np.append(k, k_cutoff)
                nk = np.append(nk, np.interp(k_cutoff, original_k, original_nk))
                nkerr = np.append(
                    nkerr,
                    np.hypot(
                        (1.0 - fraction) * original_nkerr[lower_index],
                        fraction * original_nkerr[upper_index],
                    ),
                )

        if k.size < 2:
            raise ValueError("At least two finite, positive-k points are required for integration.")

        integrand = prefactor * np.power(k, k_power) * nk
        trapezoid_weights = np.empty_like(k)
        trapezoid_weights[0] = (k[1] - k[0]) / 2.0
        trapezoid_weights[-1] = (k[-1] - k[-2]) / 2.0
        trapezoid_weights[1:-1] = (k[2:] - k[:-2]) / 2.0
        integration_weights = prefactor * np.power(k, k_power) * trapezoid_weights
        finite_errors = np.isfinite(nkerr)
        error = np.sqrt(np.sum(np.square(integration_weights[finite_errors] * nkerr[finite_errors])))
        return float(np.trapz(integrand, k)), float(error)

    def ellsq(self, n0_bar=1500, R=21., L=42., zeta=1.9) -> Tuple[float, float]:
        """Return the square of the coherence length and its uncertainty using Gevorg's procedure."""
        V = np.pi * R**2 * L # um^3
        ell0 = V**(1/3) / (zeta**(2/3)-1)**0.5 # um

        ellprime = (self.n0 / (n0_bar*zeta) * V)**(1/3) # um
        ellprime_err = 1/3 * (self.n0err / self.n0) * ellprime

        ellsq = ellprime**2 / (1 - ellprime**2 / ell0**2)
        ellsq_err = ellprime_err * (2*ellprime / (1-ellprime**2/ell0**2) + 2*ellprime**3 / (ell0**2 * (1-ellprime**2/ell0**2)**2))

        return ellsq, ellsq_err
