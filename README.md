# Momentum Distribution Analysis Pipeline

This repository contains tools to process cold-atoms momentum distributions from per-shot profile files:

- `ds_<suffix>.txt` (per-shot scalar metadata; includes `ImageNumber`, `N`, `Energy`, `ToF`, etc.)
- `k3d_<suffix>.txt` (k-axis values per shot)
- `nk3d_<suffix>.txt` (occupation values per shot)

The interactive pipeline is implemented in [momentumPipeline.py](/Users/nmasl/Desktop/AnalysisPipeline.worktrees/momentum-distribution-analysis-tool/momentumPipeline.py).

## What the pipeline does

1. **Remove bad images** with a matplotlib GUI (manual clicking + sigma-based outlier blanking).
2. **Average momentum distributions** over remaining shots per parameter group, with standard error.
   This also writes `averaged_ds_<suffix>.txt`: one row per parameter combination,
   with mean and standard-error columns for every non-`ImageNumber` ds field.
3. **Rescale detuned data** against non-detuned references using a GUI.
4. **Patch TOFs/detunings** into final single momentum distributions using user-selected validity ranges.

No original input files are modified.

## Example usage

```python
from runParameters import RunParameters
from momentumPipeline import run_full_pipeline

# Example: image numbers present in your dataset
run_numbers = list(range(1, 401))

# Define your experimental parameter pattern
variable_names = ["detuning", "ToF", "ShakeHeatTime", "Fesh_evapfinal"]
operators = [".", "x", "x"]  # detuning and ToF zipped; then crossed with others
values = {
    "detuning": [12, 10, 8, 6],
    "ToF": [20, 24, 28, 32],
    "ShakeHeatTime": [2.5, 3.0, 3.5],
    "Fesh_evapfinal": [3.933, 3.952, 3.955],
}

run_params = RunParameters(
    run_numbers=run_numbers,
    variable_names=variable_names,
    operators=operators,
    values=values,
)

pipeline = run_full_pipeline(
    data_directory="/absolute/path/to/profiles",
    data_suffix="2026-02-16_SFridayRelaxData",
    run_parameters=run_params,
    output_directory="/absolute/path/to/output",
    sort_parameter="ToF",         # order groups in bad-image GUI
    detuning_parameter="detuning",
    tof_parameter="ToF",
    non_detuned_value=12,         # default non-detuned value
    detuning_activation_times={-55: 300},  # use -55 data only at waittime >= 300
    two_d=False,                  # set True for k2d/nk2d input files
)
```

`detuning_activation_times` is optional and maps each detuning to the earliest
time at which it should contribute. The time parameter defaults to `"waittime"`
when available (otherwise `sort_parameter`); override it with
`activation_time_parameter` when your time variable has another name.
Detunings omitted from the mapping activate at the lowest scheduled time. Before
a detuning activates, it is excluded from final patched profiles. Once an
activated detuned profile has the same non-detuning parameters as a reference
profile, that non-detuned profile is excluded from the final result instead.
All data remain visible in the bad-image, rescaling, and validity-range GUIs;
series excluded from the final result are labelled accordingly. Averaged profile
files are retained for every parameter combination.

### Combining acquisitions from multiple runs

Pass `runs` instead of the three single-run input arguments. Each `PipelineRun`
has its own `RunParameters`, profile directory and filename suffix:

```python
from momentumPipeline import PipelineRun, run_full_pipeline

# Keep each acquisition's actual image numbers and parameter schedule.
original_params = RunParameters(
    [1, 2, 3, 4], ["waittime", "detuning", "ToF"], ["x", "."],
    {"waittime": [0, 100], "detuning": [6], "ToF": [60]},
)
extra_params = RunParameters(
    [1, 2, 3, 4], ["waittime", "detuning", "ToF"], ["x", "."],
    {"waittime": [100, 200], "detuning": [6], "ToF": [60]},
)
pipeline = run_full_pipeline(
    runs=[
        PipelineRun("/path/to/original/Profiles", "original_suffix", original_params,
                    name="original"),
        PipelineRun("/path/to/extra/Profiles", "extra_suffix", extra_params,
                    name="extra"),
    ],
    output_directory="/path/to/combined/Output",
    sort_parameter="waittime",
    non_detuned_value=6,
)
```

All runs must declare the same parameter names; their values, variable order,
operators, image numbers and repetition counts may differ. Shots with identical
values for **all** parameters are pooled before averaging. In this example,
waittime 100 combines both acquisitions, while 0 and 200 remain separate.
Means and standard errors use individual shots, so runs with more retained
shots contribute proportionally more. Only images present in a run's files
and its `RunParameters` schedule contribute. Missing images do not shift the
schedule. Exclusions and all subsequent processing settings apply to every run.
Missing ds columns in one run contribute no values to that column's statistics.

In `runs` mode, shots receive consecutive IDs starting at 1, ordered by the
input run list and then original image number. This avoids collisions when
different acquisitions reuse image numbers. The bad-image GUI and its cutoff
use these combined IDs; hovering over the N/Energy axes shows the source run
and original image number in the coordinate readout. `shot_sources.json` maps
every ID back to its run, directory, suffix and original image number; averaged
profile manifests also record their contributing sources. Names are optional
(default `run_1`, `run_2`, etc.) and must be unique.

All results go to the one `output_directory`, including
`averaged_ds_combined.txt`. Combined `blanks.json` files include the ID mapping
and can only be reused with the same source mapping. After adding/reordering
runs or changing the loaded shot set, select blanks again; single-run blanks
cannot be applied to combined IDs. Existing single-run calls remain supported
and retain their original image numbers and output names.

`analysis/GPEB/s17.py` shows this setup with an optional extra acquisition.
The same `runs` argument is available on `MomentumDistributionPipeline` for
step-by-step processing.

### Excluding bad parameter combinations

Pass `excluded_parameter_combinations` to `run_full_pipeline` or
`MomentumDistributionPipeline` to skip unwanted data throughout the GUIs,
averaging, rescaling, and final patching:

```python
excluded_parameter_combinations=[{"detuning": -58}]
```

Each dictionary matches all of its fields together. For example,
`[{"detuning": -58, "ToF": 120}, {"a": 100, "waittime": 300}]`
excludes shots matching either combination. Unspecified parameters can have any
value. Unknown parameter names and empty dictionaries raise an error.
Keep the original `RunParameters` values and run numbers unchanged so image
numbers retain their correct parameter assignments. Excluded detuned groups
also no longer replace their non-detuned reference groups.

When reusing an output directory, manifests describe the current run; old CSV
files from earlier runs may still exist and should not be treated as current results.

After completion, `output_directory` will contain:

- `blanks.json`
- `averaged_profiles/`
- `averaged_ds_<suffix>.txt`
- `detuning_rescale_factors.json`
- `patch_validity_ranges.json`
- `final_profiles/`

Rescaled profiles are retained in memory for patching but are not saved separately:
they are fully reproducible by applying `detuning_rescale_factors.json` to the
matching files in `averaged_profiles/`.

Final profiles are constructed on logarithmic k bins. Within each bin, each
Each `(ToF, detuning, ...)` configuration contributes one local log-log estimate at the bin
centre; those estimates are then inverse-variance weighted together. The final
CSV therefore uses `n_combinations` for its fourth column rather than
`n_shots`.

## Step-by-step (non-wrapper) usage

If you want to run each stage explicitly:

```python
from momentumPipeline import MomentumDistributionPipeline

pipeline = MomentumDistributionPipeline(
    data_directory="/absolute/path/to/profiles",
    data_suffix="2026-02-16_SFridayRelaxData",
    run_parameters=run_params,
    output_directory="/absolute/path/to/output",
    sort_parameter="ToF",
    detuning_parameter="detuning",
    tof_parameter="ToF",
    non_detuned_value=12,
)

pipeline.remove_bad_images()
pipeline.compute_averaged_momentum_distributions()
pipeline.rescale_detuned_images()
pipeline.select_patch_validity_ranges()
pipeline.patch_tofs_and_detunings()
```

## Reusing completed stages

Pass any saved JSON result back to the pipeline to skip its corresponding GUI.
The remaining stages run normally, so the final profiles can be regenerated from
the saved analysis choices.

```python
pipeline = MomentumDistributionPipeline(
    data_directory="/absolute/path/to/profiles",
    data_suffix="2026-02-16_SFridayRelaxData",
    run_parameters=run_params,
    output_directory="/absolute/path/to/new-output",
    detuning_parameter="detuning",
    tof_parameter="ToF",
    blanks_json="/previous-output/blanks.json",
    detuning_rescale_factors_json="/previous-output/detuning_rescale_factors.json",
    patch_validity_ranges_json="/previous-output/patch_validity_ranges.json",
)

# These calls use the supplied files, without opening a GUI for those stages.
pipeline.remove_bad_images()
pipeline.compute_averaged_momentum_distributions()
pipeline.rescale_detuned_images()
pipeline.select_patch_validity_ranges()
pipeline.patch_tofs_and_detunings()
```

Each JSON argument is optional. For example, pass only `blanks_json` to reuse
image choices while choosing new detuning factors and patch ranges.

While the patch-range GUI is open, use **Load ranges** to inspect and reuse the
matching combination ranges from any saved `patch_validity_ranges.json` file.
Ranges are clipped to the k range available in the current dataset; entries for
combinations not present in the current dataset are skipped.

## Interactive GUI controls

All three review windows support **Save and close**, which writes that stage's
JSON result and continues the pipeline; closing the window normally does the
same. **Exit without saving** continues the pipeline with the current choices
held in memory, without writing a JSON file. **Stop pipeline** closes without
saving and ends the current Python run before the next stage starts.

### Bad-image filtering

- **Lower σ** / **Upper σ** set the automatic outlier thresholds for atom
  number and energy. **Max image** excludes every later image.
- The **Momentum plot ranges** fields affect only the view. Enter limits and
  choose **Apply ranges**, or use **Reset ranges** for automatic axes.
- Click a trace or point to toggle that image's inclusion. **Unblank group**
  removes manual decisions for the displayed group; **Reset all** restores the
  initial filtering settings.
- **Previous** / **Next** switch groups. **Load JSON** imports a saved
  `blanks.json` for inspection or reuse.

### Detuning rescaling

- Enter a trial factor in **Scale factor** to preview it. **Save scale & next**
  records that factor for the current detuning and advances to the next pair.
- **Previous** / **Next** switch comparison pairs. The x/y limit boxes and
  **Apply limits** control the view; **Reset limits** restores automatic axes.
- **Load JSON** imports `detuning_rescale_factors.json` values for matching
  detunings.

### Patch ranges

- The top buttons select the active `(ToF, detuning, ...)` configuration. Any
  non-swept experimental parameters (for example `ZeroaV`) are included, so a
  calibration setting has independent validity limits. Use the k sliders
  or their numeric fields to choose its validity range.
- The global **Box radius (μm)** is used by the per-endpoint box-radii controls:
  `k = 0.613526 × box_radii × box_radius_um / ToF`.
- The y-limit fields set manual plot limits; **Auto y limits** restores the
  data-driven defaults. **Previous set** / **Next set** switch non-TOF,
  non-detuning parameter sets.
- **Load ranges** imports matching entries from `patch_validity_ranges.json`.

## Notes

- GUI interaction requires a matplotlib backend that supports windows/events.
- For deterministic re-runs, keep the saved JSON files in the output directory.
- If your detuning/TOF parameter names differ, pass the correct names via `detuning_parameter` and `tof_parameter`.
