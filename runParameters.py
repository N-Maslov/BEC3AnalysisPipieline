from itertools import product
from typing import List, Dict, Any, Sequence


class RunParameters:
    """
    Example usage:
    run_numbers = ODProcessed.inums

    # Get cicero run info
    variable_names = ["ShakeHeatTime", "Fesh_evapfinal", "WaitTimeTotal"]
    operators = [".", "x"]

    values = {
        "ShakeHeatTime": [3.5, 3.0, 2.5],
        "Fesh_evapfinal": [3.933, 3.952, 3.955],
        "WaitTimeTotal": [25, 60000,10000,50000,20000,40000,30000]
    }

    runs = RunParameters(
        run_numbers=run_numbers,
        variable_names=variable_names,
        operators=operators,
        values=values,
    )

    for inum in run_numbers:
        runparams = runs[inum]
        """
    def __init__(
        self,
        run_numbers: Sequence[int],
        variable_names: List[str],
        operators: List[str],
        values: Dict[str, List[Any]],
    ):
        self.run_numbers = list(run_numbers)
        self.variable_names = variable_names
        self.operators = operators
        self.values = values
        self.periodicity = None  # Will be set in _build_runs()

        self._validate()
        self.runs = self._build_runs()

    def _validate(self):
        if len(self.operators) != len(self.variable_names) - 1:
            raise ValueError(
                "Number of operators must be one less than number of variables."
            )

        for op in self.operators:
            if op not in (".", "x"):
                raise ValueError(f"Invalid operator: {op}")

        for name in self.variable_names:
            if name not in self.values:
                raise ValueError(f"Missing values for variable '{name}'")

    def _build_dot_blocks(self):
        """
        Split variables into dot-connected blocks.
        Format: [[var1, var2], [var3], [var4, var5, var6], ...]
        """
        blocks = []
        current_block = [self.variable_names[0]]

        for op, name in zip(self.operators, self.variable_names[1:]):
            if op == ".":
                current_block.append(name)
            else:  # cross
                blocks.append(current_block)
                current_block = [name]

        blocks.append(current_block)
        return blocks

    def _build_runs(self) -> Dict[int, Dict[str, Any]]:
        dot_blocks = self._build_dot_blocks()

        # Build zipped combinations inside each dot block
        block_combinations = []
        for block in dot_blocks:
            block_values = [self.values[name] for name in block]

            lengths = {len(v) for v in block_values}
            if len(lengths) != 1:
                raise ValueError(
                    f"All dotted variables must have same length: {block}"
                )

            zipped = [
                dict(zip(block, vals))
                for vals in zip(*block_values)
            ]
            block_combinations.append(zipped)

        # Cartesian product between blocks
        all_combinations = []
        for combo in product(*block_combinations):
            merged = {}
            for d in combo:
                merged.update(d)
            all_combinations.append(merged)

        self.periodicity = len(all_combinations)

        if len(self.run_numbers) < len(all_combinations):
            # Not enough runs for parameters to repeat
            return dict(zip(self.run_numbers[:len(all_combinations)], all_combinations))
        else:
            # Repeat parameter sets to fill all runs
            repeated_combos = (all_combinations * ((len(self.run_numbers) // len(all_combinations)) + 1))[:len(self.run_numbers)]
            return dict(zip(self.run_numbers, repeated_combos))
        
    def filter(self, **criteria) -> List[int]:
        filtered_inums = []
        for inum, params in self.runs.items():
            if all(params.get(key) == value for key, value in criteria.items()):
                filtered_inums.append(inum)
        return filtered_inums

    def __getitem__(self, run_number: int):
        return self.runs[run_number]

    def items(self):
        return self.runs.items()

    def __len__(self):
        return len(self.runs)