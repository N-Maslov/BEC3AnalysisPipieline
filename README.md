# Momentum Distribution Analysis Pipeline

This repository contains tools to process cold-atoms momentum distributions from per-shot profile files:

- `ds_<suffix>.txt` (per-shot scalar metadata; includes `ImageNumber`, `N`, `Energy`, `ToF`, etc.)
- `k3d_<suffix>.txt` (k-axis values per shot)
- `nk3d_<suffix>.txt` (occupation values per shot)

The interactive pipeline is implemented in [momentumPipeline.py](/Users/nmasl/Desktop/AnalysisPipeline.worktrees/momentum-distribution-analysis-tool/momentumPipeline.py).

## What the pipeline does

1. **Remove bad images** with a matplotlib GUI (manual clicking + sigma-based outlier blanking).
2. **Average momentum distributions** over remaining shots per parameter group, with standard error.
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
    two_d=False,                  # set True for k2d/nk2d input files
)
```

After completion, `output_directory` will contain:

- `blanks.json`
- `averaged_profiles/`
- `detuning_rescale_factors.json`
- `patch_validity_ranges.json`
- `final_profiles/`

Rescaled profiles are retained in memory for patching but are not saved separately:
they are fully reproducible by applying `detuning_rescale_factors.json` to the
matching files in `averaged_profiles/`.

Final profiles are constructed on logarithmic k bins. Within each bin, each
`(ToF, detuning)` combination contributes one local log-log estimate at the bin
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

## Notes

- GUI interaction requires a matplotlib backend that supports windows/events.
- For deterministic re-runs, keep the saved JSON files in the output directory.
- If your detuning/TOF parameter names differ, pass the correct names via `detuning_parameter` and `tof_parameter`.
