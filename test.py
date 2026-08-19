from momentumPipeline import MomentumDistributionPipeline
from runParameters import RunParameters

# Example: image numbers present in your dataset
run_numbers = list(range(591, 1395+1))

# Define your experimental parameter pattern
variable_names = ["waittime", "detuning", "ToF", "ZeroaV"]
operators = ["x", ".", "."]
values = {
    "waittime": [100,1600,1400,450,250,900,800,350,700,550,200,0,50,150,300,600,1000,500,1200,400],
    "detuning": [12,12,12,-55,12],
    "ToF": [30,70,120,120,80],
    "ZeroaV": [3.74, 3.74, 3.74, 3.74, 4.19]
}

run_params = RunParameters(
    run_numbers=run_numbers,
    variable_names=variable_names,
    operators=operators,
    values=values,
)
pipeline = MomentumDistributionPipeline(
    data_directory="/Volumes/amopzh/ZH_Shared/BEC 3/Analysis/Projects/GPEB/Processing/2026-08-18/profiles",
    data_suffix="2026-08-18_S10a",
    run_parameters=run_params,
    output_directory="./output",
    sort_parameter="waittime",
    detuning_parameter="detuning",
    tof_parameter="ToF",
    non_detuned_value=12,
)

pipeline.remove_bad_images()
pipeline.compute_averaged_momentum_distributions()
pipeline.rescale_detuned_images()
#pipeline.select_patch_validity_ranges()
#pipeline.patch_tofs_and_detunings()