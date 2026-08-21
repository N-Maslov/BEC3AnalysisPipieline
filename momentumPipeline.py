import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider, TextBox

from imageProcessing import ImageProcessing
from runParameters import RunParameters


@dataclass(frozen=True)
class ParameterGroup:
    params: Tuple[Tuple[str, Any], ...]
    run_numbers: Tuple[int, ...]

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.params)

    @property
    def key(self) -> str:
        parts = []
        for name, value in self.params:
            safe_value = str(value).replace("/", "_").replace(" ", "")
            parts.append(f"{name}={safe_value}")
        return "__".join(parts)


@dataclass
class AveragedProfile:
    group: ParameterGroup
    run_numbers: List[int]
    k: np.ndarray
    nk: np.ndarray
    stderr: np.ndarray
    n_shots_per_point: np.ndarray
    scale_factor: float = 1.0
    included_in_final: bool = True


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_json_file(path: str, description: str) -> Any:
    """Load a user-supplied pipeline result with a helpful error message."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load {description} JSON file '{path}': {exc}") from exc


def _ensure_numeric_array(val: Any) -> np.ndarray:
    """Ensure val is a 1D numpy array of floats.

    Accepts:
    - bytes or str containing comma-separated numbers
    - numpy scalar or ndarray (numeric or string)
    - python lists/tuples
    """
    # bytes/str scalar with commas
    if isinstance(val, (bytes, str)):
        s = val.decode() if isinstance(val, bytes) else val
        parts = [p for p in s.split(',') if p.strip() != '']
        return np.asarray([float(p) for p in parts], dtype=float)

    # python list/tuple
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=float)

    # numpy array or scalar
    if isinstance(val, np.ndarray):
        # numeric array
        if np.issubdtype(val.dtype, np.number):
            return val.astype(float)
        # object or string array: try astype(float) first
        try:
            return val.astype(float)
        except Exception:
            # try joining and splitting
            try:
                joined = ','.join([x.decode() if isinstance(x, bytes) else str(x) for x in val.flat])
                parts = [p for p in joined.split(',') if p.strip() != '']
                return np.asarray([float(p) for p in parts], dtype=float)
            except Exception:
                pass
        # 0-d array containing string
        if val.ndim == 0:
            return _ensure_numeric_array(val.item())

    # numpy scalar
    if isinstance(val, (np.floating, np.integer)):
        return np.asarray([float(val)], dtype=float)

    # fallback try
    try:
        return np.asarray(val, dtype=float)
    except Exception as e:
        raise ValueError(f"Cannot parse numeric array from value: {val!r}") from e


def group_run_numbers(
    run_parameters: RunParameters,
    run_numbers: Sequence[int],
    sort_parameter: Optional[str] = None,
) -> List[ParameterGroup]:
    grouped: Dict[Tuple[Tuple[str, Any], ...], List[int]] = {}
    ordered_names = list(run_parameters.variable_names)

    for run_number in run_numbers:
        params = run_parameters[run_number]
        key = tuple((name, _to_python_scalar(params[name])) for name in ordered_names)
        grouped.setdefault(key, []).append(run_number)

    groups: List[ParameterGroup] = [
        ParameterGroup(params=key, run_numbers=tuple(sorted(group_runs)))
        for key, group_runs in grouped.items()
    ]

    if sort_parameter is None:
        groups.sort(key=lambda g: tuple(v for _, v in g.params))
    else:
        groups.sort(
            key=lambda g: (
                g.as_dict().get(sort_parameter, 0),
                tuple(v for _, v in g.params),
            )
        )

    return groups


def _interp_to_reference_grid(
    reference_k: np.ndarray,
    shot_k: np.ndarray,
    shot_nk: np.ndarray,
) -> np.ndarray:
    sort_idx = np.argsort(shot_k)
    k_sorted = np.asarray(shot_k)[sort_idx]
    nk_sorted = np.asarray(shot_nk)[sort_idx]

    unique_k, unique_indices = np.unique(k_sorted, return_index=True)
    unique_nk = nk_sorted[unique_indices]

    if unique_k.size < 2:
        return np.full_like(reference_k, np.nan, dtype=float)

    interpolated = np.interp(reference_k, unique_k, unique_nk, left=np.nan, right=np.nan)
    outside = (reference_k < unique_k[0]) | (reference_k > unique_k[-1])
    interpolated[outside] = np.nan
    return interpolated


def average_profiles(
    profiles: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not profiles:
        raise ValueError("Cannot average an empty list of profiles.")

    ref_index = int(np.argmax([len(k_vals) for k_vals, _ in profiles]))
    reference_k = np.asarray(profiles[ref_index][0], dtype=float)
    matrix = np.full((len(profiles), len(reference_k)), np.nan, dtype=float)

    for idx, (k_vals, nk_vals) in enumerate(profiles):
        matrix[idx, :] = _interp_to_reference_grid(
            reference_k,
            np.asarray(k_vals, dtype=float),
            np.asarray(nk_vals, dtype=float),
        )

    counts = np.sum(np.isfinite(matrix), axis=0)
    means = np.nanmean(matrix, axis=0)

    stderr = np.zeros_like(means)
    valid_for_stderr = counts > 1
    if np.any(valid_for_stderr):
        stderr[valid_for_stderr] = (
            np.nanstd(matrix[:, valid_for_stderr], axis=0, ddof=1)
            / np.sqrt(counts[valid_for_stderr])
        )

    valid = counts > 0
    return reference_k[valid], means[valid], stderr[valid], counts[valid]


def _write_profile_csv(
    filepath: Path,
    k: np.ndarray,
    nk: np.ndarray,
    stderr: np.ndarray,
    n_shots: np.ndarray,
) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "nk", "stderr", "n_shots"])
        for row in zip(k, nk, stderr, n_shots):
            writer.writerow(row)


def _mean_and_stderr(values: Sequence[Any]) -> Tuple[float, float]:
    """Return the finite-value mean and sample standard error for a ds column."""
    numeric_values = np.asarray(values, dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    if numeric_values.size == 0:
        return float("nan"), float("nan")
    if numeric_values.size == 1:
        return float(numeric_values[0]), 0.0
    return (
        float(np.mean(numeric_values)),
        float(np.std(numeric_values, ddof=1) / np.sqrt(numeric_values.size)),
    )


class BadImageSelectionGUI:
    def __init__(
        self,
        image_processing: ImageProcessing,
        groups: Sequence[ParameterGroup],
        output_dir: Path,
        excluded_from_final_group_keys: Optional[Iterable[str]] = None,
    ):
        self.image_processing = image_processing
        self.groups = list(groups)
        self.output_dir = output_dir
        self.excluded_from_final_group_keys = set(excluded_from_final_group_keys or ())

        self.group_idx = 0
        self.manual_blanks: set[int] = set()
        # A manual include takes precedence over a sigma-based exclusion.  Keep
        # this separately from manual_blanks so clicking an auto-blanked shot
        # can restore it without changing the current sigma thresholds.
        self.manual_includes: set[int] = set()
        self.sigma_thresholds: Tuple[float, float] = (3.0, 3.0)
        self.sigma_blanks_by_group: Dict[str, set[int]] = {group.key: set() for group in self.groups}
        self.artist_to_inum: Dict[Any, int] = {}
        self.final_status_text: Optional[Any] = None
        self.k_range: Optional[Tuple[float, float]] = None
        self.log_nk_y_range: Optional[Tuple[float, float]] = None
        self.linear_nk_y_range: Optional[Tuple[float, float]] = None
        all_run_numbers = [inum for group in self.groups for inum in group.run_numbers]
        self.max_image_number = max(all_run_numbers)
        self.min_image_number = min(all_run_numbers) - 1

        # Give the controls their own left-hand column, keeping the full figure
        # height available to the 2×2 plot grid.
        self.fig = plt.figure(figsize=(20, 10))
        grid = self.fig.add_gridspec(
            2,
            3,
            left=0.07,
            right=0.97,
            bottom=0.10,
            top=0.78,
            width_ratios=(0.50, 1, 1),
            wspace=0.38,
            hspace=0.42,
        )
        self.axes = [
            self.fig.add_subplot(grid[0, 1]),
            self.fig.add_subplot(grid[0, 2]),
            self.fig.add_subplot(grid[1, 1]),
            self.fig.add_subplot(grid[1, 2]),
        ]
        self.ax_profiles, self.ax_linear_nk, self.ax_n, self.ax_energy = self.axes

        controls = grid[:, 0].get_position(self.fig)
        control_left, control_bottom = controls.x0, controls.y0
        control_width, control_height = controls.width, controls.height

        def control_axes(
            x_fraction: float,
            y_fraction: float,
            width_fraction: float,
            height_fraction: float,
        ) -> Any:
            return self.fig.add_axes([
                control_left + x_fraction * control_width,
                control_bottom + y_fraction * control_height,
                width_fraction * control_width,
                height_fraction * control_height,
            ])

        self.fig.text(
            control_left + control_width / 2,
            control_bottom + 0.93 * control_height,
            "Controls",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

        self.low_sigma_slider = Slider(
            control_axes(0.05, 0.83, 0.90, 0.040),
            "Lower σ",
            0.0,
            6.0,
            valinit=3.0,
            valstep=0.1,
        )
        self.high_sigma_slider = Slider(
            control_axes(0.05, 0.75, 0.90, 0.040),
            "Upper σ",
            0.0,
            6.0,
            valinit=3.0,
            valstep=0.1,
        )
        self.max_image_slider = Slider(
            control_axes(0.05, 0.68, 0.90, 0.040),
            "Max image",
            self.min_image_number,
            self.max_image_number,
            valinit=self.max_image_number,
            valstep=1,
        )

        self.range_panel = control_axes(0.02, 0.21, 0.96, 0.43)
        self.range_panel.set_facecolor("0.96")
        self.range_panel.set_xticks([])
        self.range_panel.set_yticks([])
        self.range_panel.set_title("Momentum plot ranges", fontsize=8, pad=4)
        self.fig.text(control_left + 0.545 * control_width, control_bottom + 0.625 * control_height, "Min", ha="center", va="center", fontsize=7)
        self.fig.text(control_left + 0.825 * control_width, control_bottom + 0.625 * control_height, "Max", ha="center", va="center", fontsize=7)

        def range_row(label: str, y_fraction: float) -> Tuple[TextBox, TextBox]:
            self.fig.text(
                control_left + 0.06 * control_width,
                control_bottom + y_fraction * control_height,
                label,
                ha="left",
                va="center",
                fontsize=7,
            )
            boxes = (
                TextBox(control_axes(0.42, y_fraction - 0.0275, 0.25, 0.055), "", initial="auto"),
                TextBox(control_axes(0.70, y_fraction - 0.0275, 0.25, 0.055), "", initial="auto"),
            )
            for box in boxes:
                box.text_disp.set_fontsize(8)
            return boxes

        self.k_min_box, self.k_max_box = range_row("k", 0.55)
        self.log_nk_y_min_box, self.log_nk_y_max_box = range_row("log n(k)", 0.465)
        self.linear_nk_y_min_box, self.linear_nk_y_max_box = range_row("linear n(k)", 0.38)
        self.btn_apply_plot_ranges = Button(control_axes(0.05, 0.245, 0.42, 0.045), "Apply ranges")
        self.btn_reset_plot_ranges = Button(control_axes(0.53, 0.245, 0.42, 0.045), "Reset ranges")

        self.btn_prev = Button(control_axes(0.05, 0.15, 0.42, 0.045), "Previous")
        self.btn_next = Button(control_axes(0.53, 0.15, 0.42, 0.045), "Next")
        self.btn_reset_group = Button(control_axes(0.05, 0.095, 0.90, 0.040), "Unblank group")
        self.btn_reset_all = Button(control_axes(0.05, 0.050, 0.90, 0.035), "Reset all")
        self.btn_save = Button(control_axes(0.05, 0.010, 0.90, 0.035), "Save and close")
        for button in (self.btn_apply_plot_ranges, self.btn_reset_plot_ranges):
            button.label.set_fontsize(7)
        for button in (
            self.btn_prev,
            self.btn_next,
            self.btn_reset_group,
            self.btn_reset_all,
            self.btn_save,
        ):
            button.label.set_fontsize(8)

        self.low_sigma_slider.on_changed(self._on_sigma_changed)
        self.high_sigma_slider.on_changed(self._on_sigma_changed)
        self.max_image_slider.on_changed(self._on_max_image_changed)
        self.btn_apply_plot_ranges.on_clicked(self._apply_plot_ranges)
        self.btn_reset_plot_ranges.on_clicked(self._reset_plot_ranges)
        self.btn_prev.on_clicked(self._prev_group)
        self.btn_next.on_clicked(self._next_group)
        self.btn_reset_group.on_clicked(self._reset_group)
        self.btn_reset_all.on_clicked(self._reset_all)
        self.btn_save.on_clicked(self._save_and_close)
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)

        self._recalculate_sigma_blanks()
        self._refresh_plot()

    def _current_group(self) -> ParameterGroup:
        return self.groups[self.group_idx]

    def _effective_blanks(self) -> List[int]:
        auto_blanks = set().union(*self.sigma_blanks_by_group.values())
        return sorted(
            self.manual_blanks
            | (auto_blanks - self.manual_includes)
            | self._cutoff_blanks()
        )

    def _cutoff_blanks(self) -> set[int]:
        """Images after the global image-number cutoff are always blanked."""
        return {
            inum
            for group in self.groups
            for inum in group.run_numbers
            if inum > self.max_image_number
        }

    def _effective_group_blanks(self, group: ParameterGroup) -> set[int]:
        """Return exclusions for a group after manual overrides are applied."""
        return self.manual_blanks | (
            self.sigma_blanks_by_group[group.key] - self.manual_includes
        ) | self._cutoff_blanks()

    def _calc_sigma_blanks(self, group: ParameterGroup, low_sigma: float, high_sigma: float) -> set[int]:
        n_values: List[float] = []
        e_values: List[float] = []
        run_numbers = list(group.run_numbers)
        for inum in run_numbers:
            calc = self.image_processing[inum]["calc"]
            n_values.append(float(calc["N"]))
            e_values.append(float(calc["Energy"]))

        n_arr = np.array(n_values, dtype=float)
        e_arr = np.array(e_values, dtype=float)

        blanks = set()
        for values in (n_arr, e_arr):
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if std <= 0:
                continue
            lower = mean - low_sigma * std
            upper = mean + high_sigma * std
            for idx, value in enumerate(values):
                if value < lower or value > upper:
                    blanks.add(run_numbers[idx])
        return blanks

    def _recalculate_sigma_blanks(self) -> None:
        low_sigma, high_sigma = self.sigma_thresholds
        self.sigma_blanks_by_group = {
            group.key: self._calc_sigma_blanks(group, low_sigma, high_sigma)
            for group in self.groups
        }

    def _refresh_plot(self) -> None:
        group = self._current_group()
        params = group.as_dict()
        group_blanks = self._effective_group_blanks(group)

        self.artist_to_inum.clear()
        for axis in self.axes:
            axis.clear()

        self.ax_profiles.set_title("Individual momentum distributions")
        self.ax_profiles.set_xlabel("k")
        self.ax_profiles.set_ylabel("nk")
        self.ax_profiles.set_xscale("log")
        self.ax_profiles.set_yscale("log")

        self.ax_linear_nk.set_title("Individual momentum distributions (linear scale)")
        self.ax_linear_nk.set_xlabel("k")
        self.ax_linear_nk.set_ylabel("nk")
        self.ax_linear_nk.set_xscale("log")
        self.ax_linear_nk.set_yscale("linear")

        self.ax_n.set_title("Atom number N")
        self.ax_n.set_xlabel("Image number")
        self.ax_n.set_ylabel("N")

        self.ax_energy.set_title("Energy")
        self.ax_energy.set_xlabel("Image number")
        self.ax_energy.set_ylabel("Energy")

        cmap = plt.cm.get_cmap("tab20", max(1, len(group.run_numbers)))
        n_values = []
        e_values = []
        x_values = []

        for idx, inum in enumerate(group.run_numbers):
            color = cmap(idx)
            data = self.image_processing[inum]
            k_vals = _ensure_numeric_array(data["k"]) 
            nk_vals = _ensure_numeric_array(data["nk"]) 
            calc = data["calc"]
            n_value = float(calc["N"])
            e_value = float(calc["Energy"])

            x_values.append(inum)
            n_values.append(n_value)
            e_values.append(e_value)

            valid_log = np.isfinite(k_vals) & np.isfinite(nk_vals) & (k_vals > 0) & (nk_vals > 0)
            valid_linear = np.isfinite(k_vals) & np.isfinite(nk_vals) & (k_vals > 0)
            alpha = 0.2 if inum in group_blanks else 0.9
            profile_line, = self.ax_profiles.plot(
                k_vals[valid_log],
                nk_vals[valid_log],
                color=color,
                alpha=alpha,
                picker=5,
            )
            linear_profile_line, = self.ax_linear_nk.plot(
                k_vals[valid_linear],
                nk_vals[valid_linear],
                color=color,
                alpha=alpha,
                picker=5,
            )
            n_point = self.ax_n.scatter([inum], [n_value], color=[color], alpha=alpha, picker=True, s=45)
            e_point = self.ax_energy.scatter([inum], [e_value], color=[color], alpha=alpha, picker=True, s=45)
            self.artist_to_inum[profile_line] = inum
            self.artist_to_inum[linear_profile_line] = inum
            self.artist_to_inum[n_point] = inum
            self.artist_to_inum[e_point] = inum

        self.ax_n.plot(x_values, n_values, color="0.6", alpha=0.4)
        self.ax_energy.plot(x_values, e_values, color="0.6", alpha=0.4)
        if self.k_range is not None:
            self.ax_profiles.set_xlim(*self.k_range)
            self.ax_linear_nk.set_xlim(*self.k_range)
        if self.log_nk_y_range is not None:
            self.ax_profiles.set_ylim(*self.log_nk_y_range)
        if self.linear_nk_y_range is not None:
            self.ax_linear_nk.set_ylim(*self.linear_nk_y_range)

        title = ", ".join(f"{k}={v}" for k, v in params.items())
        self.fig.suptitle(
            f"Group {self.group_idx + 1}/{len(self.groups)} | {title}\n"
            f"Effective blanks in this group: {sum(i in group_blanks for i in group.run_numbers)}",
            fontsize=11,
        )
        if self.final_status_text is not None:
            self.final_status_text.remove()
        self.final_status_text = None
        if group.key in self.excluded_from_final_group_keys:
            self.final_status_text = self.fig.text(
                0.5,
                0.855,
                "Not included in final patching",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="red",
            )
        self.fig.canvas.draw_idle()

    def _on_pick(self, event: Any) -> None:
        artist = event.artist
        inum = self.artist_to_inum.get(artist)
        if inum is None and isinstance(artist, Line2D):
            inum = self.artist_to_inum.get(artist)
        if inum is None:
            return
        if inum in self._cutoff_blanks():
            return
        auto_blanks = set().union(*self.sigma_blanks_by_group.values())
        if inum in self._effective_blanks():
            self.manual_blanks.discard(inum)
            if inum in auto_blanks:
                self.manual_includes.add(inum)
        else:
            self.manual_blanks.add(inum)
            self.manual_includes.discard(inum)
        self._refresh_plot()

    def _on_sigma_changed(self, _: float) -> None:
        self.sigma_thresholds = (
            float(self.low_sigma_slider.val),
            float(self.high_sigma_slider.val),
        )
        self._recalculate_sigma_blanks()
        self._refresh_plot()

    def _on_max_image_changed(self, value: float) -> None:
        self.max_image_number = int(round(value))
        self._refresh_plot()

    @staticmethod
    def _range_from_text_boxes(
        min_box: TextBox,
        max_box: TextBox,
        label: str,
        *,
        positive: bool,
    ) -> Optional[Tuple[float, float]]:
        """Read a range, with ``auto`` in both fields restoring autoscaling."""
        min_text = min_box.text.strip().lower()
        max_text = max_box.text.strip().lower()
        if min_text == max_text == "auto":
            return None
        try:
            lower = float(min_text)
            upper = float(max_text)
        except ValueError:
            raise ValueError(f"Enter both {label} limits, or use auto for both.") from None
        if upper <= lower:
            raise ValueError(f"The {label} maximum must be greater than its minimum.")
        if positive and lower <= 0:
            raise ValueError(f"The {label} minimum must be positive.")
        return lower, upper

    def _apply_plot_ranges(self, _: Any) -> None:
        """Apply all requested momentum-plot axis ranges at once."""
        try:
            self.k_range = self._range_from_text_boxes(
                self.k_min_box, self.k_max_box, "k", positive=True
            )
            self.log_nk_y_range = self._range_from_text_boxes(
                self.log_nk_y_min_box,
                self.log_nk_y_max_box,
                "log n(k)",
                positive=True,
            )
            self.linear_nk_y_range = self._range_from_text_boxes(
                self.linear_nk_y_min_box,
                self.linear_nk_y_max_box,
                "linear n(k)",
                positive=False,
            )
        except ValueError as exc:
            self.range_panel.set_title(str(exc), fontsize=7, color="red", pad=4)
            self.fig.canvas.draw_idle()
            return
        self.range_panel.set_title("Momentum plot ranges", fontsize=8, color="black", pad=4)
        self._refresh_plot()

    def _reset_plot_ranges(self, _: Any) -> None:
        self.k_range = None
        self.log_nk_y_range = None
        self.linear_nk_y_range = None
        self.k_min_box.set_val("auto")
        self.k_max_box.set_val("auto")
        self.log_nk_y_min_box.set_val("auto")
        self.log_nk_y_max_box.set_val("auto")
        self.linear_nk_y_min_box.set_val("auto")
        self.linear_nk_y_max_box.set_val("auto")
        self.range_panel.set_title("Momentum plot ranges", fontsize=8, color="black", pad=4)
        self._refresh_plot()

    def _prev_group(self, _: Any) -> None:
        self.group_idx = (self.group_idx - 1) % len(self.groups)
        self._refresh_plot()

    def _next_group(self, _: Any) -> None:
        self.group_idx = (self.group_idx + 1) % len(self.groups)
        self._refresh_plot()

    def _reset_group(self, _: Any) -> None:
        group = self._current_group()
        for inum in group.run_numbers:
            self.manual_blanks.discard(inum)
            self.manual_includes.discard(inum)
        self._refresh_plot()

    def _reset_all(self, _: Any) -> None:
        self.manual_blanks.clear()
        self.manual_includes.clear()
        self.max_image_number = max(
            inum for group in self.groups for inum in group.run_numbers
        )
        self.max_image_slider.set_val(self.max_image_number)
        self.sigma_thresholds = (3.0, 3.0)
        self.low_sigma_slider.set_val(3.0)
        self.high_sigma_slider.set_val(3.0)
        self._recalculate_sigma_blanks()
        self._refresh_plot()

    def save(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        outpath = self.output_dir / "blanks.json"
        payload = {
            "blank_image_numbers": self._effective_blanks(),
            "manual_blanks": sorted(self.manual_blanks),
            "manual_includes": sorted(self.manual_includes),
            "max_image_number": self.max_image_number,
            "sigma_thresholds": {
                "lower_sigma": self.sigma_thresholds[0],
                "upper_sigma": self.sigma_thresholds[1],
            },
        }
        outpath.write_text(json.dumps(payload, indent=2))
        return outpath

    def _save_and_close(self, _: Any) -> None:
        self.save()
        plt.close(self.fig)

    def launch(self) -> List[int]:
        plt.show()
        self.save()
        return self._effective_blanks()


class DetuningRescaleGUI:
    def __init__(
        self,
        averaged_profiles: Sequence[AveragedProfile],
        detuning_parameter: str,
        non_detuned_value: Any,
        output_dir: Path,
        sort_parameter: Optional[str] = None,
    ):
        self.detuning_parameter = detuning_parameter
        self.non_detuned_value = non_detuned_value
        self.output_dir = output_dir
        self.all_profiles = list(averaged_profiles)
        self.sort_parameter = sort_parameter

        # map profile params -> profile for lookup
        self.profile_map: Dict[Tuple[Tuple[str, Any], ...], AveragedProfile] = {
            profile.group.params: profile for profile in self.all_profiles
        }
        # pairs: list of (detuned_profile, reference_profile)
        self.pairs = self._build_pairs()
        # Saved scales are keyed by detuning value (one scale per detuning).
        self.confirmed_scales: Dict[Any, float] = {}
        self.pair_idx = 0
        self.current_scale = 1.0

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.subplots_adjust(bottom=0.31, top=0.92)
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel("k")
        self.ax.set_ylabel("nk")

        # Enter previews a scale. "Save scale & next" persists it globally for
        # this detuning and advances to the next comparison.
        self.fig.text(0.06, 0.235, "Scale factor", ha="left", va="bottom")
        self.scale_box = TextBox(self.fig.add_axes([0.06, 0.17, 0.16, 0.055]), "")
        self.btn_confirm = Button(self.fig.add_axes([0.25, 0.17, 0.18, 0.055]), "Save scale & next")
        self.btn_prev = Button(self.fig.add_axes([0.46, 0.17, 0.10, 0.055]), "Previous")
        self.btn_next = Button(self.fig.add_axes([0.58, 0.17, 0.10, 0.055]), "Next")
        self.btn_save = Button(self.fig.add_axes([0.72, 0.17, 0.20, 0.055]), "Save and close")

        # Put labels above the text boxes so they cannot be clipped by them.
        for x_position, label in ((0.06, "x min"), (0.20, "x max"), (0.34, "y min"), (0.48, "y max")):
            self.fig.text(x_position, 0.115, label, ha="left", va="bottom")
        self.xmin_box = TextBox(self.fig.add_axes([0.06, 0.055, 0.12, 0.045]), "")
        self.xmax_box = TextBox(self.fig.add_axes([0.20, 0.055, 0.12, 0.045]), "")
        self.ymin_box = TextBox(self.fig.add_axes([0.34, 0.055, 0.12, 0.045]), "")
        self.ymax_box = TextBox(self.fig.add_axes([0.48, 0.055, 0.12, 0.045]), "")
        self.btn_apply_limits = Button(self.fig.add_axes([0.63, 0.055, 0.13, 0.045]), "Apply limits")
        self.btn_reset_limits = Button(self.fig.add_axes([0.78, 0.055, 0.14, 0.045]), "Reset limits")

        # callbacks
        self.btn_confirm.on_clicked(self._confirm_scale)
        self.btn_prev.on_clicked(self._prev_pair)
        self.btn_next.on_clicked(self._next_pair)
        self.btn_save.on_clicked(self._save_and_close)
        self.scale_box.on_submit(lambda txt: self._preview_scale())
        self.btn_apply_limits.on_clicked(self._apply_limits)
        self.btn_reset_limits.on_clicked(self._reset_limits)

        # initialize default axis limits
        self._default_xlim = None
        self._default_ylim = (100.0, 1e7)
        # persistent user-specified limits (None means not set)
        self.user_xlim: Optional[Tuple[float, float]] = None
        self.user_ylim: Optional[Tuple[float, float]] = None

        self._refresh_plot()

    def _apply_limits(self, _: Any) -> None:
        # parse boxes and store as persistent user limits, then refresh
        try:
            xmin = float(self.xmin_box.text.strip()) if self.xmin_box.text.strip() else None
            xmax = float(self.xmax_box.text.strip()) if self.xmax_box.text.strip() else None
            ymin = float(self.ymin_box.text.strip()) if self.ymin_box.text.strip() else None
            ymax = float(self.ymax_box.text.strip()) if self.ymax_box.text.strip() else None
            if xmin is not None and xmax is not None:
                self.user_xlim = (xmin, xmax)
            else:
                self.user_xlim = None
            if ymin is not None and ymax is not None:
                self.user_ylim = (ymin, ymax)
            else:
                self.user_ylim = None
        except Exception:
            return
        self._refresh_plot()

    def _reset_limits(self, _: Any) -> None:
        # clear text boxes and reset to defaults
        try:
            self.xmin_box.set_val("")
            self.xmax_box.set_val("")
            self.ymin_box.set_val("")
            self.ymax_box.set_val("")
        except Exception:
            pass
        # clear persistent user limits
        self.user_xlim = None
        self.user_ylim = None
        if self._default_xlim and self._default_xlim[0] is not None:
            self.ax.set_xlim(self._default_xlim)
        if self._default_ylim and self._default_ylim[0] is not None:
            self.ax.set_ylim(self._default_ylim)
        self.fig.canvas.draw_idle()

    def _build_pairs(self) -> List[Tuple[AveragedProfile, AveragedProfile]]:
        pairs: List[Tuple[AveragedProfile, AveragedProfile]] = []
        for profile in self.all_profiles:
            params = profile.group.as_dict()
            if params.get(self.detuning_parameter) == self.non_detuned_value:
                continue

            reference_params = []
            for name, value in profile.group.params:
                if name == self.detuning_parameter:
                    reference_params.append((name, self.non_detuned_value))
                else:
                    reference_params.append((name, value))
            reference_key = tuple(reference_params)
            reference_profile = self.profile_map.get(reference_key)
            if reference_profile is None:
                continue
            pairs.append((profile, reference_profile))

        # Sort pairs by the requested sort_parameter (if provided) so user sees a sensible order
        if self.sort_parameter is not None:
            def sort_key(pair: Tuple[AveragedProfile, AveragedProfile]):
                params = pair[0].group.as_dict()
                return params.get(self.sort_parameter, 0)
            pairs.sort(key=sort_key)
        else:
            pairs.sort(key=lambda pair: pair[0].group.key)
        return pairs

    def _estimate_initial_scale(
        self,
        detuned: AveragedProfile,
        reference: AveragedProfile,
    ) -> float:
        # Estimate ratio detuned->reference on overlapping k points
        det_on_ref = _interp_to_reference_grid(reference.k, detuned.k, detuned.nk)
        valid = np.isfinite(det_on_ref) & (det_on_ref > 0) & np.isfinite(reference.nk) & (reference.nk > 0)
        if np.sum(valid) < 3:
            return 1.0
        ratio = reference.nk[valid] / det_on_ref[valid]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size == 0:
            return 1.0
        return float(np.median(ratio))

    def _initial_scale_for_detuning(self, detuning_value: Any) -> float:
        # compute initial median of estimated scales across all pairs with this detuning
        estimates = []
        for detuned, reference in self.pairs:
            params = detuned.group.as_dict()
            if params.get(self.detuning_parameter) != detuning_value:
                continue
            est = self._estimate_initial_scale(detuned, reference)
            if np.isfinite(est) and est > 0:
                estimates.append(est)
        if not estimates:
            return 1.0
        return float(np.median(estimates))

    def _current_pair(self) -> Tuple[AveragedProfile, AveragedProfile]:
        return self.pairs[self.pair_idx]

    def _refresh_plot(self, keep_preview: bool = False) -> None:
        self.ax.clear()
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel("k")
        self.ax.set_ylabel("nk")

        if not self.pairs:
            self.ax.text(0.5, 0.5, "No detuned/reference pairs found.", ha="center", va="center")
            self.fig.canvas.draw_idle()
            return

        detuned, reference = self._current_pair()
        detuning_value = detuned.group.as_dict().get(self.detuning_parameter)

        if not keep_preview:
            # Prefer a saved value for this detuning; otherwise use an estimate.
            if detuning_value in self.confirmed_scales:
                self.current_scale = self.confirmed_scales[detuning_value]
            else:
                self.current_scale = self._initial_scale_for_detuning(detuning_value)
            self.scale_box.set_val(f"{self.current_scale:.6g}")

        scaled_nk = detuned.nk * self.current_scale
        scaled_err = detuned.stderr * self.current_scale

        # set colors explicitly to avoid duplicate colors
        ref_color = "tab:blue"
        det_color = "tab:orange"
        reference_label = "Non-detuned reference"
        if not reference.included_in_final:
            reference_label += " (excluded from final)"
        self.ax.loglog(reference.k, reference.nk, "o-", ms=3, label=reference_label, color=ref_color)
        # fill reference errors if available
        try:
            if reference.stderr is not None and np.any(np.isfinite(reference.stderr)):
                ref_err = reference.stderr
                self.ax.fill_between(reference.k, reference.nk - ref_err, reference.nk + ref_err, color=ref_color, alpha=0.15)
        except Exception:
            pass
        detuned_label = "Detuned (scaled)"
        if not detuned.included_in_final:
            detuned_label += " (excluded from final)"
        self.ax.loglog(detuned.k, scaled_nk, "o-", ms=3, label=detuned_label, color=det_color)
        self.ax.fill_between(detuned.k, scaled_nk - scaled_err, scaled_nk + scaled_err, color=det_color, alpha=0.2)

        # initialize default axis limits if not set
        if self._default_xlim is None:
            try:
                xmin = min(np.nanmin(reference.k), np.nanmin(detuned.k))
                xmax = max(np.nanmax(reference.k), np.nanmax(detuned.k))
                self._default_xlim = (xmin, xmax)
            except Exception:
                self._default_xlim = (None, None)
        if self._default_ylim is None:
            try:
                ymin = min(np.nanmin(reference.nk), np.nanmin(scaled_nk))
                ymax = max(np.nanmax(reference.nk), np.nanmax(scaled_nk))
                self._default_ylim = (ymin, ymax)
            except Exception:
                self._default_ylim = (None, None)

        # apply any user-specified limits in text boxes if present
        try:
            xmin_txt = self.xmin_box.text.strip()
            xmax_txt = self.xmax_box.text.strip()
            if xmin_txt and xmax_txt:
                self.ax.set_xlim(float(xmin_txt), float(xmax_txt))
            else:
                if self._default_xlim[0] is not None:
                    self.ax.set_xlim(self._default_xlim)
        except Exception:
            pass
        try:
            ymin_txt = self.ymin_box.text.strip()
            ymax_txt = self.ymax_box.text.strip()
            if ymin_txt and ymax_txt:
                self.ax.set_ylim(float(ymin_txt), float(ymax_txt))
            else:
                if self._default_ylim[0] is not None:
                    self.ax.set_ylim(self._default_ylim)
        except Exception:
            pass

        title = ", ".join(f"{k}={v}" for k, v in detuned.group.as_dict().items())
        self.ax.set_title(f"Pair {self.pair_idx + 1}/{len(self.pairs)} | {title}")
        self.ax.legend(loc="best")
        self.fig.canvas.draw_idle()

    def _preview_scale(self) -> None:
        if not self.pairs:
            return
        try:
            text = self.scale_box.text.strip()
            value = float(text)
        except Exception:
            return
        if value <= 0:
            raise ValueError("Scale factor must be positive.")
        self.current_scale = value
        self._refresh_plot(keep_preview=True)

    def _confirm_scale(self, _: Any) -> None:
        if not self.pairs:
            return
        self._preview_scale()
        detuned, _ = self._current_pair()
        detuning_value = detuned.group.as_dict().get(self.detuning_parameter)
        self.confirmed_scales[detuning_value] = self.current_scale
        if self.pair_idx < len(self.pairs) - 1:
            self.pair_idx += 1
        self._refresh_plot()

    def _prev_pair(self, _: Any) -> None:
        if not self.pairs:
            return
        self.pair_idx = (self.pair_idx - 1) % len(self.pairs)
        self._refresh_plot()

    def _next_pair(self, _: Any) -> None:
        if not self.pairs:
            return
        self.pair_idx = (self.pair_idx + 1) % len(self.pairs)
        self._refresh_plot()

    def _detuning_scale_factors(self) -> Dict[Any, float]:
        # Return a mapping detuning_value -> factor, shared across all other parameters.
        factors: Dict[Any, float] = {}
        detuning_values = {
            profile.group.as_dict().get(self.detuning_parameter)
            for profile in self.all_profiles
        }
        for detuning_value in detuning_values:
            if detuning_value == self.non_detuned_value:
                factor = 1.0
            elif detuning_value in self.confirmed_scales:
                factor = float(self.confirmed_scales[detuning_value])
            else:
                matching = [
                    pair
                    for pair in self.pairs
                    if pair[0].group.as_dict().get(self.detuning_parameter) == detuning_value
                ]
                if matching:
                    factor = float(self._estimate_initial_scale(matching[0][0], matching[0][1]))
                else:
                    factor = 1.0
            factors[detuning_value] = factor
        return factors

    def save(self) -> Tuple[Path, Dict[Any, float]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        factors = self._detuning_scale_factors()
        outpath = self.output_dir / "detuning_rescale_factors.json"
        json_payload = {str(detuning): factor for detuning, factor in factors.items()}
        outpath.write_text(json.dumps(json_payload, indent=2))
        return outpath, factors

    def _save_and_close(self, _: Any) -> None:
        self.save()
        plt.close(self.fig)

    def launch(self) -> Dict[Any, float]:
        plt.show()
        _, factors = self.save()
        return factors


class PatchRangesGUI:
    def __init__(
        self,
        patch_sets: Sequence[Tuple[Tuple[Tuple[str, Any], ...], List[AveragedProfile]]],
        tof_parameter: str,
        detuning_parameter: str,
        output_dir: Path,
    ):
        self.patch_sets = list(patch_sets)
        self.tof_parameter = tof_parameter
        self.detuning_parameter = detuning_parameter
        self.output_dir = output_dir
        self.patch_idx = 0
        self.save_on_close = True

        self.combo_bounds = self._compute_combo_bounds()
        self.validity_ranges: Dict[str, Tuple[float, float]] = dict(self.combo_bounds)
        # Select combinations by increasing detuning, then decreasing time of flight.
        # Use two stable sorts so the requested descending secondary ordering is
        # retained without needing to negate parameter values.
        self.combo_keys = sorted(
            sorted(
                self.combo_bounds,
                key=lambda key: self._sortable_value(self.combo_values[key][1]),
                reverse=True,
            ),
            key=lambda key: self._sortable_value(self.combo_values[key][0]),
        )
        colour_map = plt.get_cmap("tab20", max(len(self.combo_keys), 1))
        self.combo_colors = {
            combo_key: colour_map(index)
            for index, combo_key in enumerate(self.combo_keys)
        }
        self.selected_combo = self.combo_keys[0] if self.combo_keys else ""
        self.nk_y_limits: Optional[Tuple[float, float]] = None
        self.k2nk_y_limits: Optional[Tuple[float, float]] = None

        self.fig, (self.ax, self.ax_k2nk) = plt.subplots(1, 2, figsize=(18, 9))
        self.fig.subplots_adjust(bottom=0.38, left=0.06, right=0.88, top=0.72, wspace=0.28)
        self.fig.text(0.06, 0.955, "Select (ToF, detuning)", weight="bold", color="#1f2937")
        self._build_combo_selector()
        self.patch_set_title = self.fig.text(0.42, 0.735, "", ha="center", fontsize=14, weight="bold")
        self.status_text = self.fig.text(0.66, 0.205, "", color="tab:green")
        self.shared_legend: Optional[Any] = None
        self.box_radius_um = 21.0

        self.slider_ax_min = self.fig.add_axes([0.06, 0.27, 0.50, 0.025])
        self.slider_ax_max = self.fig.add_axes([0.06, 0.225, 0.50, 0.025])
        self.k_min_box = TextBox(self.fig.add_axes([0.59, 0.262, 0.07, 0.04]), "", initial="")
        self.k_max_box = TextBox(self.fig.add_axes([0.59, 0.217, 0.07, 0.04]), "", initial="")
        self.fig.text(0.68, 0.280, "or set to", color="#374151")
        self.fig.text(0.68, 0.235, "or set to", color="#374151")
        self.k_min_box_radii_box = TextBox(
            self.fig.add_axes([0.76, 0.262, 0.05, 0.04]), "", initial="3"
        )
        self.k_max_box_radii_box = TextBox(
            self.fig.add_axes([0.76, 0.217, 0.05, 0.04]), "", initial="7"
        )
        self.fig.text(0.82, 0.280, "box radii", color="#374151")
        self.fig.text(0.82, 0.235, "box radii", color="#374151")
        self.btn_set_k_min_from_radii = Button(
            self.fig.add_axes([0.89, 0.262, 0.05, 0.04]), "Set"
        )
        self.btn_set_k_max_from_radii = Button(
            self.fig.add_axes([0.89, 0.217, 0.05, 0.04]), "Set"
        )
        self.slider_min: Optional[Slider] = None
        self.slider_max: Optional[Slider] = None
        self.k_min_box.on_submit(self._on_k_min_text_submit)
        self.k_max_box.on_submit(self._on_k_max_text_submit)
        self.k_min_box_radii_box.on_submit(self._on_k_min_box_radii_submit)
        self.k_max_box_radii_box.on_submit(self._on_k_max_box_radii_submit)
        self.btn_set_k_min_from_radii.on_clicked(self._set_k_min_from_box_radii)
        self.btn_set_k_max_from_radii.on_clicked(self._set_k_max_from_box_radii)
        self._build_sliders_for_combo(self.selected_combo)

        self.fig.text(0.06, 0.20, r"$n_k$ y limits (log)", color="#374151")
        self.fig.text(0.33, 0.20, r"$k^2 n_k$ y limits (linear)", color="#374151")
        self.nk_y_min_box = TextBox(self.fig.add_axes([0.06, 0.157, 0.12, 0.03]), "min", initial="auto")
        self.nk_y_max_box = TextBox(self.fig.add_axes([0.06, 0.117, 0.12, 0.03]), "max", initial="auto")
        self.k2nk_y_min_box = TextBox(self.fig.add_axes([0.33, 0.157, 0.12, 0.03]), "min", initial="auto")
        self.k2nk_y_max_box = TextBox(self.fig.add_axes([0.33, 0.117, 0.12, 0.03]), "max", initial="auto")
        self.btn_apply_y_limits = Button(self.fig.add_axes([0.50, 0.157, 0.12, 0.03]), "Apply y limits")
        self.btn_auto_y_limits = Button(self.fig.add_axes([0.50, 0.117, 0.12, 0.03]), "Auto y limits")
        self.fig.text(0.66, 0.175, "Box radius (μm)", color="#374151")
        self.box_radius_box = TextBox(self.fig.add_axes([0.78, 0.157, 0.08, 0.03]), "", initial="21")
        self.btn_set_box_radius = Button(self.fig.add_axes([0.88, 0.157, 0.06, 0.03]), "Set")
        self.box_radius_box.on_submit(self._set_box_radius)
        self.btn_set_box_radius.on_clicked(self._set_box_radius)
        self.nk_y_min_box.on_submit(self._apply_y_limits)
        self.nk_y_max_box.on_submit(self._apply_y_limits)
        self.k2nk_y_min_box.on_submit(self._apply_y_limits)
        self.k2nk_y_max_box.on_submit(self._apply_y_limits)
        self.btn_apply_y_limits.on_clicked(self._apply_y_limits)
        self.btn_auto_y_limits.on_clicked(self._reset_y_limits)
        self.fig.canvas.mpl_connect("button_press_event", self._clear_auto_y_limit_field)

        self.btn_prev = Button(self.fig.add_axes([0.06, 0.02, 0.14, 0.045]), "Previous set")
        self.btn_next = Button(self.fig.add_axes([0.22, 0.02, 0.14, 0.045]), "Next set")
        self.btn_load = Button(self.fig.add_axes([0.38, 0.02, 0.14, 0.045]), "Load ranges")
        self.btn_exit = Button(self.fig.add_axes([0.54, 0.02, 0.14, 0.045]), "Exit without saving")
        self.btn_save = Button(self.fig.add_axes([0.70, 0.02, 0.14, 0.045]), "Save and close")
        self.btn_prev.on_clicked(self._prev_set)
        self.btn_next.on_clicked(self._next_set)
        self.btn_load.on_clicked(self._load_ranges)
        self.btn_exit.on_clicked(self._exit_without_saving)
        self.btn_save.on_clicked(self._save_and_close)

        self._refresh_plot()

    def _build_combo_selector(self) -> None:
        """Create a compact grid of large buttons for selecting a series."""
        self.combo_buttons: Dict[str, Button] = {}
        if not self.combo_keys:
            return

        columns = min(4, max(1, ceil(sqrt(len(self.combo_keys)))))
        rows = ceil(len(self.combo_keys) / columns)
        left, bottom, width, height = 0.06, 0.775, 0.88, 0.15
        gap_x, gap_y = 0.01, 0.012
        button_width = (width - gap_x * (columns - 1)) / columns
        button_height = (height - gap_y * (rows - 1)) / rows
        for index, combo_key in enumerate(self.combo_keys):
            row, column = divmod(index, columns)
            x = left + column * (button_width + gap_x)
            y = bottom + (rows - 1 - row) * (button_height + gap_y)
            button = Button(
                self.fig.add_axes([x, y, button_width, button_height]),
                combo_key,
                color="#f1f5f9",
                hovercolor="#dbeafe",
            )
            button.on_clicked(lambda _, key=combo_key: self._on_combo_selected(key))
            self.combo_buttons[combo_key] = button
        self._update_combo_button_styles()

    def _update_combo_button_styles(self) -> None:
        for combo_key, button in self.combo_buttons.items():
            selected = combo_key == self.selected_combo
            # Button restores ``color`` after mouse events, so update that
            # property too; changing only the axes facecolor flashes briefly.
            button.color = "#1e3a8a" if selected else "#f1f5f9"
            button.hovercolor = "#1e40af" if selected else "#dbeafe"
            button.ax.set_facecolor(button.color)
            button.label.set_color("white" if selected else "#1f2937")

    def _compute_combo_bounds(self) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}
        self.combo_values: Dict[str, Tuple[Any, Any]] = {}
        for _, profiles in self.patch_sets:
            for profile in profiles:
                params = profile.group.as_dict()
                combo_key = self._combo_key(params)
                self.combo_values[combo_key] = (
                    params[self.detuning_parameter],
                    params[self.tof_parameter],
                )
                kmin = float(np.nanmin(profile.k))
                kmax = float(np.nanmax(profile.k))
                if combo_key not in bounds:
                    bounds[combo_key] = (kmin, kmax)
                else:
                    low, high = bounds[combo_key]
                    bounds[combo_key] = (min(low, kmin), max(high, kmax))
        return bounds

    @staticmethod
    def _sortable_value(value: Any) -> Tuple[int, Any]:
        """Sort numerical parameter values numerically, with a text fallback."""
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))

    def _combo_key(self, params: Dict[str, Any]) -> str:
        return f"{self.tof_parameter}={params[self.tof_parameter]}, {self.detuning_parameter}={params[self.detuning_parameter]}"

    def _build_sliders_for_combo(self, combo_key: str) -> None:
        self.slider_ax_min.clear()
        self.slider_ax_max.clear()
        if combo_key == "":
            return
        low, high = self.combo_bounds[combo_key]
        current_low, current_high = self.validity_ranges[combo_key]

        if low <= 0 or high <= 0:
            raise ValueError("Patch validity ranges require strictly positive k values.")
        log_low, log_high = np.log10((low, high))

        self.slider_min = Slider(
            self.slider_ax_min,
            "k min",
            log_low,
            log_high,
            valinit=np.log10(current_low),
        )
        self.slider_max = Slider(
            self.slider_ax_max,
            "k max",
            log_low,
            log_high,
            valinit=np.log10(current_high),
        )
        # The editable fields beside the sliders show the physical k values;
        # hide Slider's duplicate log-coordinate readout.
        self.slider_min.valtext.set_visible(False)
        self.slider_max.valtext.set_visible(False)
        self._update_slider_value_labels()
        self.slider_min.on_changed(self._on_slider_change)
        self.slider_max.on_changed(self._on_slider_change)

    def _update_slider_value_labels(self) -> None:
        """Show physical k values while the slider positions use log10(k)."""
        if self.slider_min is not None:
            self.k_min_box.set_val(f"{10 ** self.slider_min.val:.6g}")
        if self.slider_max is not None:
            self.k_max_box.set_val(f"{10 ** self.slider_max.val:.6g}")

    def _set_k_limit_from_text(self, value_text: str, is_minimum: bool) -> None:
        if self.selected_combo == "" or self.slider_min is None or self.slider_max is None:
            return
        try:
            value = float(value_text)
        except ValueError:
            value = float("nan")
        bound_low, bound_high = self.combo_bounds[self.selected_combo]
        other_value = float(10 ** (self.slider_max.val if is_minimum else self.slider_min.val))
        valid_order = value <= other_value if is_minimum else value >= other_value
        if not np.isfinite(value) or not bound_low <= value <= bound_high or not valid_order:
            self.status_text.set_text(
                f"Enter a k value from {bound_low:.6g} to {bound_high:.6g}, without crossing the other limit."
            )
            self.status_text.set_color("tab:red")
            self._update_slider_value_labels()
            self.fig.canvas.draw_idle()
            return
        slider = self.slider_min if is_minimum else self.slider_max
        slider.set_val(np.log10(value))
        self.status_text.set_text("")

    def _on_k_min_text_submit(self, value_text: str) -> None:
        self._set_k_limit_from_text(value_text, is_minimum=True)

    def _on_k_max_text_submit(self, value_text: str) -> None:
        self._set_k_limit_from_text(value_text, is_minimum=False)

    def _set_box_radius(self, _: Any) -> None:
        try:
            box_radius = float(self.box_radius_box.text)
        except ValueError:
            box_radius = float("nan")
        if not np.isfinite(box_radius) or box_radius <= 0:
            self.status_text.set_text("Box radius must be a positive number of μm.")
            self.status_text.set_color("tab:red")
            self.fig.canvas.draw_idle()
            return
        self.box_radius_um = box_radius
        self.box_radius_box.set_val(f"{box_radius:.6g}")
        self.status_text.set_text(f"Box radius set to {box_radius:.6g} μm.")
        self.status_text.set_color("tab:green")
        self.fig.canvas.draw_idle()

    def _set_k_limit_from_box_radii(self, radii_text: str, is_minimum: bool) -> None:
        if self.selected_combo == "":
            return
        try:
            box_radii = float(radii_text)
            tof = float(self.combo_values[self.selected_combo][1])
        except (KeyError, TypeError, ValueError):
            box_radii = float("nan")
            tof = float("nan")
        if not np.isfinite(box_radii) or box_radii <= 0 or not np.isfinite(tof) or tof <= 0:
            self.status_text.set_text("Box radii and ToF must both be positive numbers.")
            self.status_text.set_color("tab:red")
            self.fig.canvas.draw_idle()
            return
        k_value = 0.613526 * box_radii * self.box_radius_um / tof
        self._set_k_limit_from_text(f"{k_value:.16g}", is_minimum)

    def _on_k_min_box_radii_submit(self, radii_text: str) -> None:
        self._set_k_limit_from_box_radii(radii_text, is_minimum=True)

    def _on_k_max_box_radii_submit(self, radii_text: str) -> None:
        self._set_k_limit_from_box_radii(radii_text, is_minimum=False)

    def _set_k_min_from_box_radii(self, _: Any) -> None:
        self._on_k_min_box_radii_submit(self.k_min_box_radii_box.text)

    def _set_k_max_from_box_radii(self, _: Any) -> None:
        self._on_k_max_box_radii_submit(self.k_max_box_radii_box.text)

    def _on_slider_change(self, _: float) -> None:
        if self.selected_combo == "" or self.slider_min is None or self.slider_max is None:
            return
        low = float(10 ** self.slider_min.val)
        high = float(10 ** self.slider_max.val)
        if low > high:
            low, high = high, low
        # Slider positions are stored as log10(k).  Converting an untouched
        # endpoint back with 10**x can differ from the original k value by a
        # few floating-point bits, excluding the first/last sample from an
        # otherwise inclusive range.
        bound_low, bound_high = self.combo_bounds[self.selected_combo]
        if np.isclose(low, bound_low, rtol=1e-12, atol=0.0):
            low = bound_low
        if np.isclose(high, bound_high, rtol=1e-12, atol=0.0):
            high = bound_high
        self.validity_ranges[self.selected_combo] = (low, high)
        self._update_slider_value_labels()
        self._refresh_plot()

    def _on_combo_selected(self, combo_key: str) -> None:
        self.selected_combo = combo_key
        self._update_combo_button_styles()
        self._build_sliders_for_combo(combo_key)
        self._refresh_plot()

    @staticmethod
    def _read_y_limits(minimum: TextBox, maximum: TextBox, log_scale: bool) -> Optional[Tuple[float, float]]:
        values = (minimum.text.strip().lower(), maximum.text.strip().lower())
        if values in (("", ""), ("auto", "auto")):
            return None
        if "" in values or "auto" in values:
            raise ValueError("Enter both limits, or use 'auto' for both.")
        try:
            low, high = (float(value) for value in values)
        except ValueError as exc:
            raise ValueError("Axis limits must be numeric.") from exc
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError("Axis limits must be finite, with minimum less than maximum.")
        if log_scale and low <= 0:
            raise ValueError("The log-scale n_k plot requires a positive y minimum.")
        return low, high

    def _apply_y_limits(self, _: Any) -> None:
        try:
            self.nk_y_limits = self._read_y_limits(self.nk_y_min_box, self.nk_y_max_box, log_scale=True)
            self.k2nk_y_limits = self._read_y_limits(self.k2nk_y_min_box, self.k2nk_y_max_box, log_scale=False)
        except ValueError as exc:
            self.status_text.set_text(f"Could not apply y limits: {exc}")
            self.status_text.set_color("tab:red")
            self.fig.canvas.draw_idle()
            return
        self.status_text.set_text("Applied y limits." if self.nk_y_limits or self.k2nk_y_limits else "Using automatic y limits.")
        self.status_text.set_color("tab:green")
        self._refresh_plot()

    def _clear_auto_y_limit_field(self, event: Any) -> None:
        """Let users type over the automatic placeholder without deleting it."""
        for box in (
            self.nk_y_min_box,
            self.nk_y_max_box,
            self.k2nk_y_min_box,
            self.k2nk_y_max_box,
        ):
            if event.inaxes is box.ax and box.text.strip().lower() == "auto":
                box.set_val("")
                return

    def _reset_y_limits(self, _: Any) -> None:
        self.nk_y_limits = None
        self.k2nk_y_limits = None
        self.nk_y_min_box.set_val("auto")
        self.nk_y_max_box.set_val("auto")
        self.k2nk_y_min_box.set_val("auto")
        self.k2nk_y_max_box.set_val("auto")
        self.status_text.set_text("Using automatic y limits.")
        self.status_text.set_color("tab:green")
        self._refresh_plot()

    def _import_validity_ranges(self, path: str) -> Tuple[int, int]:
        """Load matching ranges from a saved patch-validity-ranges JSON file."""
        payload = _load_json_file(path, "patch validity ranges")
        if not isinstance(payload, dict):
            raise ValueError("Patch-ranges JSON must contain a mapping of combinations to k ranges.")

        imported = 0
        skipped = 0
        for combo_key, values in payload.items():
            if combo_key not in self.combo_bounds:
                skipped += 1
                continue
            try:
                range_low = float(values["k_min"])
                range_high = float(values["k_max"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Each patch range must contain numeric 'k_min' and 'k_max' values."
                ) from exc
            if not np.isfinite(range_low) or not np.isfinite(range_high) or range_low <= 0 or range_high <= 0:
                raise ValueError("Patch range limits must be finite, strictly positive numbers.")
            if range_low > range_high:
                raise ValueError("Each patch range's k_min must be less than or equal to k_max.")

            available_low, available_high = self.combo_bounds[combo_key]
            range_low = max(range_low, available_low)
            range_high = min(range_high, available_high)
            if range_low > range_high:
                skipped += 1
                continue
            self.validity_ranges[combo_key] = (range_low, range_high)
            imported += 1
        return imported, skipped

    def _choose_ranges_file(self) -> str:
        """Open a native file chooser without disturbing Matplotlib's event loop."""
        if sys.platform == "darwin":
            def choose_file(default_location: str) -> str:
                result = subprocess.run(
                    [
                        "osascript",
                        "-e",
                        "POSIX path of (choose file with prompt "
                        '"Load patch validity ranges" default location '
                        f"(POSIX file {json.dumps(default_location)}))",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    if "User canceled" in result.stderr:
                        return ""
                    raise RuntimeError(result.stderr.strip() or "The file picker could not be opened.")
                return result.stdout.strip()

            try:
                # Do not filter by a Finder type: its JSON UTI recognition can
                # incorrectly disable valid .json files on some macOS versions.
                return choose_file(str(self.output_dir.resolve()))
            except RuntimeError:
                return choose_file(str(Path.home() / "Documents"))

        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        try:
            return filedialog.askopenfilename(
                parent=root,
                title="Load patch validity ranges",
                initialdir=str(self.output_dir.resolve()),
                filetypes=[("JSON files", "*.json"), ("All files", "*")],
            )
        finally:
            root.destroy()

    def _load_ranges(self, _: Any) -> None:
        """Choose and import saved ranges without closing the review GUI."""
        try:
            path = self._choose_ranges_file()
            if not path:
                return

            imported, skipped = self._import_validity_ranges(path)
        except (OSError, RuntimeError, ValueError) as exc:
            self.status_text.set_text(f"Could not load ranges: {exc}")
            self.status_text.set_color("tab:red")
            self.fig.canvas.draw_idle()
            return

        self._build_sliders_for_combo(self.selected_combo)
        self._refresh_plot()
        message = f"Loaded {imported} range{'s' if imported != 1 else ''}."
        if skipped:
            message += f" Skipped {skipped} unmatched or out-of-range entr{'ies' if skipped != 1 else 'y'}."
        self.status_text.set_text(message)
        self.status_text.set_color("tab:green")
        self.fig.canvas.draw_idle()

    def _current_patch_set(self) -> Tuple[Tuple[Tuple[str, Any], ...], List[AveragedProfile]]:
        return self.patch_sets[self.patch_idx]

    @staticmethod
    def _auto_log_y_limits(values: Sequence[np.ndarray]) -> Optional[Tuple[float, float]]:
        """Calculate log-scale limits from data values, deliberately excluding errors."""
        positive = [array[np.isfinite(array) & (array > 0)] for array in values]
        positive = [array for array in positive if array.size]
        if not positive:
            return None
        low = float(min(np.min(array) for array in positive))
        high = float(max(np.max(array) for array in positive))
        if np.isclose(low, high):
            return low / 2.0, high * 2.0
        padding = (high / low) ** 0.06
        return low / padding, high * padding

    @staticmethod
    def _auto_linear_y_limits(values: Sequence[np.ndarray]) -> Optional[Tuple[float, float]]:
        """Calculate linear-scale limits from data values, deliberately excluding errors."""
        finite = [array[np.isfinite(array)] for array in values]
        finite = [array for array in finite if array.size]
        if not finite:
            return None
        low = float(min(np.min(array) for array in finite))
        high = float(max(np.max(array) for array in finite))
        padding = 0.06 * (high - low)
        if padding == 0:
            padding = max(abs(low), 1.0) * 0.06
        return low - padding, high + padding

    def _apply_y_axis_limits(
        self,
        nk_values: Sequence[np.ndarray],
        k2nk_values: Sequence[np.ndarray],
    ) -> None:
        nk_limits = self.nk_y_limits or self._auto_log_y_limits(nk_values)
        k2nk_limits = self.k2nk_y_limits or self._auto_linear_y_limits(k2nk_values)
        if nk_limits is not None:
            self.ax.set_ylim(nk_limits)
        if k2nk_limits is not None:
            self.ax_k2nk.set_ylim(k2nk_limits)

    def _refresh_plot(self) -> None:
        if self.shared_legend is not None:
            self.shared_legend.remove()
            self.shared_legend = None
        self.ax.clear()
        self.ax_k2nk.clear()
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel("k")
        self.ax.set_ylabel("nk (rescaled)")
        self.ax_k2nk.set_xlabel("k")
        self.ax_k2nk.set_ylabel(r"$k^2 n_k$ (rescaled)")

        if not self.patch_sets:
            self.patch_set_title.set_text("No data for patching")
            self.ax.text(0.5, 0.5, "No data for patching.", ha="center", va="center")
            self.ax_k2nk.text(0.5, 0.5, "No data for patching.", ha="center", va="center")
            self.fig.canvas.draw_idle()
            return

        other_params, profiles = self._current_patch_set()
        # Draw unselected profiles first, then redraw the selected profile last.
        # This gives it the highest z-order even where curves overlap.
        ordered_profiles = sorted(
            profiles,
            key=lambda profile: self._combo_key(profile.group.as_dict()) == self.selected_combo,
        )
        nk_values: List[np.ndarray] = []
        k2nk_values: List[np.ndarray] = []
        excluded_legend_labels: set[str] = set()
        for profile in ordered_profiles:
            params = profile.group.as_dict()
            combo_key = self._combo_key(params)
            low, high = self.validity_ranges[combo_key]
            linewidth = 2.8 if combo_key == self.selected_combo else 1.4
            zorder = 3 if combo_key == self.selected_combo else 2
            in_valid_range = (profile.k >= low) & (profile.k <= high)
            if np.any(in_valid_range):
                included_in_final = profile.included_in_final
                label = f"({params[self.tof_parameter]}, {params[self.detuning_parameter]})"
                if not included_in_final:
                    excluded_legend_labels.add(label)
                k_values = profile.k[in_valid_range]
                nk_values_for_profile = profile.nk[in_valid_range]
                errors = np.abs(profile.stderr[in_valid_range])
                k2nk_values_for_profile = k_values ** 2 * nk_values_for_profile
                nk_values.append(nk_values_for_profile)
                k2nk_values.append(k2nk_values_for_profile)
                self.ax.errorbar(
                    k_values,
                    nk_values_for_profile,
                    yerr=errors,
                    fmt=".-" if included_in_final else ".--",
                    linewidth=linewidth,
                    elinewidth=max(0.6, linewidth * 0.5),
                    capsize=2,
                    label=label,
                    color=self.combo_colors[combo_key],
                    zorder=zorder,
                    alpha=1.0 if included_in_final else 0.35,
                )
                self.ax_k2nk.errorbar(
                    k_values,
                    k2nk_values_for_profile,
                    yerr=k_values ** 2 * errors,
                    fmt=".-" if included_in_final else ".--",
                    linewidth=linewidth,
                    elinewidth=max(0.6, linewidth * 0.5),
                    capsize=2,
                    label=label,
                    color=self.combo_colors[combo_key],
                    zorder=zorder,
                    alpha=1.0 if included_in_final else 0.35,
                )

        other_text = ", ".join(f"{k}={v}" for k, v in other_params)
        title = f"Patch set {self.patch_idx + 1}/{len(self.patch_sets)}"
        if other_text:
            title += f" | {other_text}"
        self.patch_set_title.set_text(title)
        self._apply_y_axis_limits(nk_values, k2nk_values)
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.shared_legend = self.fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(0.89, 0.56),
                fontsize=8,
                title="(ToF, detuning)",
                title_fontsize=8,
            )
            for text in self.shared_legend.get_texts():
                if text.get_text() in excluded_legend_labels:
                    text.set_color("tab:red")
        self.fig.canvas.draw_idle()

    def _prev_set(self, _: Any) -> None:
        if not self.patch_sets:
            return
        self.patch_idx = (self.patch_idx - 1) % len(self.patch_sets)
        self._refresh_plot()

    def _next_set(self, _: Any) -> None:
        if not self.patch_sets:
            return
        self.patch_idx = (self.patch_idx + 1) % len(self.patch_sets)
        self._refresh_plot()

    def save(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        outpath = self.output_dir / "patch_validity_ranges.json"
        payload = {
            key: {"k_min": values[0], "k_max": values[1]}
            for key, values in self.validity_ranges.items()
        }
        outpath.write_text(json.dumps(payload, indent=2))
        return outpath

    def _save_and_close(self, _: Any) -> None:
        self.save()
        plt.close(self.fig)

    def _exit_without_saving(self, _: Any) -> None:
        self.save_on_close = False
        plt.close(self.fig)

    def launch(self) -> Dict[str, Tuple[float, float]]:
        plt.show()
        if self.save_on_close:
            self.save()
        return dict(self.validity_ranges)


def _group_for_patching(
    profiles: Iterable[AveragedProfile],
    tof_parameter: str,
    detuning_parameter: str,
) -> List[Tuple[Tuple[Tuple[str, Any], ...], List[AveragedProfile]]]:
    grouped: Dict[Tuple[Tuple[str, Any], ...], List[AveragedProfile]] = {}
    for profile in profiles:
        other_params = tuple(
            (name, value)
            for name, value in profile.group.params
            if name not in (tof_parameter, detuning_parameter)
        )
        grouped.setdefault(other_params, []).append(profile)

    out = []
    for key, grouped_profiles in grouped.items():
        grouped_profiles.sort(
            key=lambda p: (
                p.group.as_dict()[tof_parameter],
                p.group.as_dict()[detuning_parameter],
            )
        )
        out.append((key, grouped_profiles))
    out.sort(key=lambda item: item[0])
    return out


def _patch_profiles(
    profiles: Sequence[AveragedProfile],
    validity_ranges: Dict[str, Tuple[float, float]],
    tof_parameter: str,
    detuning_parameter: str,
    bins_per_decade: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_k: List[float] = []
    all_nk: List[float] = []
    all_err: List[float] = []
    all_combos: List[str] = []

    for profile in profiles:
        params = profile.group.as_dict()
        combo_key = f"{tof_parameter}={params[tof_parameter]}, {detuning_parameter}={params[detuning_parameter]}"
        k_low, k_high = validity_ranges[combo_key]
        valid = (
            np.isfinite(profile.k)
            & np.isfinite(profile.nk)
            & np.isfinite(profile.stderr)
            & (profile.k >= k_low)
            & (profile.k <= k_high)
            & (profile.k > 0)
        )
        all_k.extend(profile.k[valid].tolist())
        all_nk.extend(profile.nk[valid].tolist())
        all_err.extend(profile.stderr[valid].tolist())
        all_combos.extend([combo_key] * int(np.sum(valid)))

    if not all_k:
        raise ValueError("No valid points available after applying patch validity ranges.")

    k_arr = np.asarray(all_k, dtype=float)
    nk_arr = np.asarray(all_nk, dtype=float)
    err_arr = np.asarray(all_err, dtype=float)
    combo_arr = np.asarray(all_combos, dtype=object)

    log_min = np.log10(np.min(k_arr))
    log_max = np.log10(np.max(k_arr))
    num_bins = max(20, int(np.ceil((log_max - log_min) * bins_per_decade)))
    edges = np.logspace(log_min, log_max, num_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    # np.digitize assigns a point exactly on the upper edge to one past the
    # final bin; retain it in the final bin instead of dropping it.
    bin_ids = np.minimum(np.digitize(k_arr, edges) - 1, num_bins - 1)
    valid_bins = (bin_ids >= 0) & (bin_ids < num_bins)
    bin_ids = bin_ids[valid_bins]
    nk_arr = nk_arr[valid_bins]
    err_arr = err_arr[valid_bins]
    combo_arr = combo_arr[valid_bins]

    out_k: List[float] = []
    out_nk: List[float] = []
    out_err: List[float] = []
    out_count: List[int] = []

    for bin_idx in range(num_bins):
        in_bin = bin_ids == bin_idx
        if not np.any(in_bin):
            continue
        # First reduce every (ToF, detuning) combination to one estimate, then
        # combine those estimates at the second stage.
        combo_means: List[float] = []
        combo_errors: List[float] = []
        for combo_key in np.unique(combo_arr[in_bin]):
            from_combo = in_bin & (combo_arr == combo_key)
            combo_nk = nk_arr[from_combo]
            combo_err = err_arr[from_combo]
            positive_error = combo_err > 0
            if np.any(positive_error):
                combo_weights = 1.0 / np.square(combo_err[positive_error])
                combo_means.append(
                    float(
                        np.sum(combo_weights * combo_nk[positive_error])
                        / np.sum(combo_weights)
                    )
                )
                combo_errors.append(float(np.sqrt(1.0 / np.sum(combo_weights))))
            else:
                combo_means.append(float(np.mean(combo_nk)))
                combo_errors.append(
                    float(np.std(combo_nk, ddof=1) / np.sqrt(combo_nk.size))
                    if combo_nk.size > 1
                    else 0.0
                )

        combo_means_arr = np.asarray(combo_means)
        combo_errors_arr = np.asarray(combo_errors)
        positive_combo_error = combo_errors_arr > 0
        if np.any(positive_combo_error):
            weights = 1.0 / np.square(combo_errors_arr[positive_combo_error])
            weighted_mean = float(
                np.sum(weights * combo_means_arr[positive_combo_error]) / np.sum(weights)
            )
            combined_err = float(np.sqrt(1.0 / np.sum(weights)))
        else:
            weighted_mean = float(np.mean(combo_means_arr))
            combined_err = (
                float(np.std(combo_means_arr, ddof=1) / np.sqrt(combo_means_arr.size))
                if combo_means_arr.size > 1
                else 0.0
            )

        out_k.append(float(centers[bin_idx]))
        out_nk.append(float(weighted_mean))
        out_err.append(float(combined_err))
        out_count.append(int(np.sum(in_bin)))

    return (
        np.array(out_k, dtype=float),
        np.array(out_nk, dtype=float),
        np.array(out_err, dtype=float),
        np.array(out_count, dtype=int),
    )


class MomentumDistributionPipeline:
    def __init__(
        self,
        data_directory: str,
        data_suffix: str,
        run_parameters: RunParameters,
        output_directory: str,
        *,
        sort_parameter: Optional[str] = None,
        detuning_parameter: str = "detuning",
        tof_parameter: str = "ToF",
        non_detuned_value: Any = 12,
        detuning_activation_times: Optional[Dict[Any, float]] = None,
        activation_time_parameter: Optional[str] = None,
        two_d: bool = False,
        blanks_json: Optional[str] = None,
        detuning_rescale_factors_json: Optional[str] = None,
        patch_validity_ranges_json: Optional[str] = None,
    ):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.detuning_parameter = detuning_parameter
        self.tof_parameter = tof_parameter
        self.non_detuned_value = non_detuned_value
        self.sort_parameter = sort_parameter
        self.activation_time_parameter = activation_time_parameter
        self.detuning_activation_times = self._validate_activation_times(
            detuning_activation_times
        )

        self.image_processing = ImageProcessing(data_directory, data_suffix, twoD=two_d)
        self.run_parameters = run_parameters
        self.groups = group_run_numbers(
            run_parameters=self.run_parameters,
            run_numbers=self.image_processing.inums,
            sort_parameter=sort_parameter,
        )
        if self.activation_time_parameter is None:
            self.activation_time_parameter = (
                "waittime"
                if "waittime" in self.run_parameters.variable_names
                else sort_parameter or "waittime"
            )
        self.default_detuning_activation_time = self._lowest_activation_time()
        # Retained for callers that inspect the pipeline state.  All groups are
        # shown and processed; this list identifies groups used in final patches.
        self.active_groups = [
            group for group in self.groups if self._group_is_active(group)
        ]

        self.blanks: List[int] = []
        self.averaged_profiles: List[AveragedProfile] = []
        self.rescaled_profiles: List[AveragedProfile] = []
        self.patch_ranges: Dict[str, Tuple[float, float]] = {}
        self.blanks_json = blanks_json
        self.detuning_rescale_factors_json = detuning_rescale_factors_json
        self.patch_validity_ranges_json = patch_validity_ranges_json
        self.detuning_scale_factors: Optional[Dict[str, float]] = None

        if blanks_json is not None:
            blanks_payload = _load_json_file(blanks_json, "blanks")
            blank_numbers = blanks_payload.get("blank_image_numbers") if isinstance(blanks_payload, dict) else None
            if not isinstance(blank_numbers, list):
                raise ValueError("Blanks JSON must contain a 'blank_image_numbers' list.")
            self.blanks = sorted({int(image_number) for image_number in blank_numbers})

        if detuning_rescale_factors_json is not None:
            factors_payload = _load_json_file(
                detuning_rescale_factors_json,
                "detuning rescale factors",
            )
            if not isinstance(factors_payload, dict):
                raise ValueError("Detuning-rescale JSON must contain a mapping of detuning values to scale factors.")
            self.detuning_scale_factors = {
                str(detuning): float(factor)
                for detuning, factor in factors_payload.items()
            }

        if patch_validity_ranges_json is not None:
            ranges_payload = _load_json_file(patch_validity_ranges_json, "patch validity ranges")
            if not isinstance(ranges_payload, dict):
                raise ValueError("Patch-ranges JSON must contain a mapping of combinations to k ranges.")
            try:
                self.patch_ranges = {
                    str(combo_key): (float(values["k_min"]), float(values["k_max"]))
                    for combo_key, values in ranges_payload.items()
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Each patch range must contain numeric 'k_min' and 'k_max' values."
                ) from exc

    @staticmethod
    def _validate_activation_times(
        activation_times: Optional[Dict[Any, float]],
    ) -> Dict[Any, float]:
        if activation_times is None:
            return {}
        if not isinstance(activation_times, dict):
            raise ValueError("detuning_activation_times must be a mapping of detunings to times.")
        try:
            return {
                detuning: float(activation_time)
                for detuning, activation_time in activation_times.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("Every detuning activation time must be numeric.") from exc

    def _lowest_activation_time(self) -> Optional[float]:
        """Return the earliest scheduled time, if this dataset has a time parameter."""
        time_values = []
        for group in self.groups:
            value = group.as_dict().get(self.activation_time_parameter)
            if value is None:
                continue
            try:
                time_values.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Activation time parameter '{self.activation_time_parameter}' must be numeric."
                ) from exc
        if not time_values:
            if self.detuning_activation_times:
                raise ValueError(
                    f"Activation time parameter '{self.activation_time_parameter}' was not found in the run parameters."
                )
            return None
        return min(time_values)

    def _activation_time_for_detuning(self, detuning_value: Any) -> Optional[float]:
        """Get a detuning's requested activation time, defaulting to the earliest time."""
        for configured_detuning, activation_time in self.detuning_activation_times.items():
            if configured_detuning == detuning_value or str(configured_detuning) == str(detuning_value):
                return activation_time
        return self.default_detuning_activation_time

    def _detuned_group_is_active(self, group: ParameterGroup) -> bool:
        """Whether a detuned group has reached its activation time."""
        params = group.as_dict()
        activation_time = self._activation_time_for_detuning(
            params.get(self.detuning_parameter)
        )
        # No time parameter is needed when no activation cutoffs were requested.
        if activation_time is None:
            return True
        try:
            group_time = float(params[self.activation_time_parameter])
        except KeyError as exc:
            raise ValueError(
                f"Activation time parameter '{self.activation_time_parameter}' was not found in a parameter group."
            ) from exc
        return group_time >= activation_time

    def _matching_detuned_group_is_active(self, group: ParameterGroup) -> bool:
        """Whether an active detuned counterpart replaces this reference group."""
        reference_params = group.as_dict()
        for candidate in self.groups:
            candidate_params = candidate.as_dict()
            if candidate_params.get(self.detuning_parameter) == self.non_detuned_value:
                continue
            if all(
                candidate_params.get(name) == value
                for name, value in reference_params.items()
                if name != self.detuning_parameter
            ) and self._detuned_group_is_active(candidate):
                return True
        return False

    def _group_is_active(self, group: ParameterGroup) -> bool:
        """Whether a group contributes to a final patched momentum profile."""
        if group.as_dict().get(self.detuning_parameter) != self.non_detuned_value:
            return self._detuned_group_is_active(group)
        return not self._matching_detuned_group_is_active(group)

    def remove_bad_images(self) -> List[int]:
        if self.blanks_json is not None:
            return list(self.blanks)
        gui = BadImageSelectionGUI(
            image_processing=self.image_processing,
            groups=self.groups,
            output_dir=self.output_directory,
            excluded_from_final_group_keys={
                group.key for group in self.groups if not self._group_is_active(group)
            },
        )
        self.blanks = gui.launch()
        return self.blanks

    def compute_averaged_momentum_distributions(self) -> List[AveragedProfile]:
        self.write_averaged_ds()
        averaged_dir = self.output_directory / "averaged_profiles"
        averaged_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        averaged_profiles: List[AveragedProfile] = []

        blank_set = set(self.blanks)
        for group in self.groups:
            valid_runs = [inum for inum in group.run_numbers if inum not in blank_set]
            if not valid_runs:
                continue

            profiles = []
            for inum in valid_runs:
                shot = self.image_processing[inum]
                profiles.append((_ensure_numeric_array(shot["k"]), _ensure_numeric_array(shot["nk"])))

            k_vals, nk_vals, stderr_vals, n_counts = average_profiles(profiles)
            averaged_profile = AveragedProfile(
                group=group,
                run_numbers=valid_runs,
                k=k_vals,
                nk=nk_vals,
                stderr=stderr_vals,
                n_shots_per_point=n_counts,
                included_in_final=self._group_is_active(group),
            )
            averaged_profiles.append(averaged_profile)

            file_name = f"{group.key}.csv"
            file_path = averaged_dir / file_name
            _write_profile_csv(file_path, k_vals, nk_vals, stderr_vals, n_counts)
            manifest.append(
                {
                    "group": group.as_dict(),
                    "group_key": group.key,
                    "run_numbers": valid_runs,
                    "file": file_name,
                }
            )

        (averaged_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self.averaged_profiles = averaged_profiles
        return averaged_profiles

    def write_averaged_ds(self) -> Path:
        """Write ds metadata averaged over each non-blank parameter combination.

        The output replaces the per-shot ``ImageNumber`` index with the ordered
        parameter columns from ``RunParameters``.  Each remaining source ds
        column is represented by paired ``_mean`` and ``_stderr`` columns.
        Standard errors use the sample standard deviation (``ddof=1``) and are
        zero for a combination containing one valid shot.
        """
        calc_data = self.image_processing.calc_data
        field_names = list(calc_data.dtype.names or ())
        if "ImageNumber" not in field_names:
            raise ValueError("The ds file must contain an 'ImageNumber' column.")

        value_fields = [name for name in field_names if name != "ImageNumber"]
        parameter_fields = list(self.run_parameters.variable_names)
        output_path = self.output_directory / f"averaged_ds_{self.image_processing.suffix}.txt"
        blank_set = set(self.blanks)

        # Index ds rows by image number so the grouping established from the
        # parameter schedule can be applied even if ds rows are not ordered.
        row_by_image_number = {}
        for ds_row in np.atleast_1d(calc_data):
            try:
                image_number = float(ds_row["ImageNumber"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(image_number):
                continue
            row_by_image_number[int(image_number)] = ds_row

        header = parameter_fields + [
            column_name
            for field_name in value_fields
            for column_name in (f"{field_name}_mean", f"{field_name}_stderr")
        ]
        with output_path.open("w", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(header)
            for group in self.groups:
                valid_rows = [
                    row_by_image_number[image_number]
                    for image_number in group.run_numbers
                    if image_number not in blank_set and image_number in row_by_image_number
                ]
                if not valid_rows:
                    continue

                row = [group.as_dict()[name] for name in parameter_fields]
                for field_name in value_fields:
                    mean, stderr = _mean_and_stderr(
                        [ds_row[field_name] for ds_row in valid_rows]
                    )
                    row.extend((mean, stderr))
                writer.writerow(row)

        return output_path

    def rescale_detuned_images(self) -> List[AveragedProfile]:
        if not self.averaged_profiles:
            raise ValueError("No averaged profiles. Run compute_averaged_momentum_distributions() first.")

        if self.detuning_scale_factors is None:
            rescale_gui = DetuningRescaleGUI(
                averaged_profiles=self.averaged_profiles,
                detuning_parameter=self.detuning_parameter,
                non_detuned_value=self.non_detuned_value,
                output_dir=self.output_directory,
                sort_parameter=self.sort_parameter,
            )
            detuning_scale_factors: Dict[Any, float] = rescale_gui.launch()
        else:
            detuning_scale_factors = self.detuning_scale_factors

        rescaled_profiles: List[AveragedProfile] = []
        for profile in self.averaged_profiles:
            detuning_value = profile.group.as_dict().get(self.detuning_parameter)
            factor = float(
                detuning_scale_factors.get(
                    detuning_value,
                    detuning_scale_factors.get(str(detuning_value), 1.0),
                )
            )
            scaled_profile = AveragedProfile(
                group=profile.group,
                run_numbers=list(profile.run_numbers),
                k=np.copy(profile.k),
                nk=np.copy(profile.nk) * factor,
                stderr=np.copy(profile.stderr) * factor,
                n_shots_per_point=np.copy(profile.n_shots_per_point),
                scale_factor=factor,
                included_in_final=profile.included_in_final,
            )
            rescaled_profiles.append(scaled_profile)

        self.rescaled_profiles = rescaled_profiles
        return rescaled_profiles

    def select_patch_validity_ranges(self) -> Dict[str, Tuple[float, float]]:
        if not self.rescaled_profiles:
            raise ValueError("No rescaled profiles. Run rescale_detuned_images() first.")
        if self.patch_validity_ranges_json is not None:
            return dict(self.patch_ranges)
        patch_sets = _group_for_patching(
            self.rescaled_profiles,
            tof_parameter=self.tof_parameter,
            detuning_parameter=self.detuning_parameter,
        )
        gui = PatchRangesGUI(
            patch_sets=patch_sets,
            tof_parameter=self.tof_parameter,
            detuning_parameter=self.detuning_parameter,
            output_dir=self.output_directory,
        )
        self.patch_ranges = gui.launch()
        return self.patch_ranges

    def patch_tofs_and_detunings(self) -> Path:
        if not self.rescaled_profiles:
            raise ValueError("No rescaled profiles. Run rescale_detuned_images() first.")
        if not self.patch_ranges:
            raise ValueError("Patch validity ranges not set. Run select_patch_validity_ranges() first.")

        final_dir = self.output_directory / "final_profiles"
        final_dir.mkdir(parents=True, exist_ok=True)
        patch_sets = _group_for_patching(
            self.rescaled_profiles,
            tof_parameter=self.tof_parameter,
            detuning_parameter=self.detuning_parameter,
        )
        manifest = []

        for other_params, profiles in patch_sets:
            final_profiles = [profile for profile in profiles if profile.included_in_final]
            if not final_profiles:
                continue
            k_vals, nk_vals, stderr_vals, n_counts = _patch_profiles(
                profiles=final_profiles,
                validity_ranges=self.patch_ranges,
                tof_parameter=self.tof_parameter,
                detuning_parameter=self.detuning_parameter,
            )
            key_text = "__".join(f"{k}={str(v).replace('/', '_')}" for k, v in other_params)
            file_name = f"{key_text if key_text else 'all_params'}.csv"
            _write_profile_csv(final_dir / file_name, k_vals, nk_vals, stderr_vals, n_counts)
            manifest.append(
                {
                    "other_parameters": dict(other_params),
                    "file": file_name,
                    "components": [profile.group.as_dict() for profile in final_profiles],
                }
            )

        (final_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return final_dir


def run_full_pipeline(
    data_directory: str,
    data_suffix: str,
    run_parameters: RunParameters,
    output_directory: str,
    *,
    sort_parameter: Optional[str] = None,
    detuning_parameter: str = "detuning",
    tof_parameter: str = "ToF",
    non_detuned_value: Any = 12,
    detuning_activation_times: Optional[Dict[Any, float]] = None,
    activation_time_parameter: Optional[str] = None,
    two_d: bool = False,
    blanks_json: Optional[str] = None,
    detuning_rescale_factors_json: Optional[str] = None,
    patch_validity_ranges_json: Optional[str] = None,
) -> MomentumDistributionPipeline:
    pipeline = MomentumDistributionPipeline(
        data_directory=data_directory,
        data_suffix=data_suffix,
        run_parameters=run_parameters,
        output_directory=output_directory,
        sort_parameter=sort_parameter,
        detuning_parameter=detuning_parameter,
        tof_parameter=tof_parameter,
        non_detuned_value=non_detuned_value,
        detuning_activation_times=detuning_activation_times,
        activation_time_parameter=activation_time_parameter,
        two_d=two_d,
        blanks_json=blanks_json,
        detuning_rescale_factors_json=detuning_rescale_factors_json,
        patch_validity_ranges_json=patch_validity_ranges_json,
    )
    pipeline.remove_bad_images()
    pipeline.compute_averaged_momentum_distributions()
    pipeline.rescale_detuned_images()
    pipeline.select_patch_validity_ranges()
    pipeline.patch_tofs_and_detunings()
    return pipeline
