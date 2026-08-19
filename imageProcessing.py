import numpy as np
from typing import Dict, Any

class ImageProcessing:
    """
    Container for k, nk, and calc parameters indexed by image number.
    Allows access like: params[inum][property], where property is "k", "nk", or "calc", and inum is integer.
        (calc_values are indexed by:
        ImageNumber,ToF,SubROIRadius,BecEx,N,Energy,T,N0,Nth,betaMu,CtrX,CtrY,BgOffset,CtrOD,Alpha,Amplitude,Residue,ImagingIntensity,Fudge,XW,YW)
    Also index by attribute: params.k_data[inum], params.nk_data[inum], params.calc_data[field], where field as above.
    Example usage:
        dir = '/Volumes/amopzh/ZH_Shared/BEC 3/Analysis/Projects/Elastic3body/Processing/2026-02-16/test'
        suffix = '2026-02-16_SFridayRelaxData'
        params = ImageProcessing(dir, suffix)

        nk = params[3]["nk"]
        k  = params[3]["k"]
        calc = params[3]["calc"]

        print("nk:", nk)
        print("k:", k)
        print("calc:", calc["ImageNumber"], calc["ToF"], calc["SubROIRadius"])
    """

    def __init__(self, directory: str, suffix: str, blanks=[], twoD=False):
        self.directory = directory
        self.suffix = suffix
        self._twoD = twoD
        self._blanks = blanks

        self._load_data()
        inums_list = []
        for i in self.calc_data["ImageNumber"].tolist():
            try:
                if int(i) not in self._blanks:
                    inums_list.append(int(i))
            except:
                pass
        #self.inums = [int(i) for i in self.calc_data["ImageNumber"].tolist() if int(i) not in self._blanks]
        self.inums = inums_list
        self._init_image = self.inums[0]

        # Cache: image_number -> dict
        self._cache: Dict[int, Dict[str, Any]] = {}

    # ---------- public API ----------

    def __getitem__(self, inum: int) -> Dict[str, Any]:
        if not isinstance(inum, int):
            raise TypeError("Image number must be an integer")

        if inum not in self._cache:
            self._cache[inum] = self._get_from_inum(inum)

        return self._cache[inum]
    
    def __len__(self):
        return len(self.inums)
    
    def get_averaged_nk(self, inums: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inums = self._remove_blanks(inums)
        nk_values = []
        for inum in inums:
            try:
                nk_values.append(self[inum]["nk"])
            except KeyError:
                print(f"Warning: No data for image number {inum}, skipping.")
        
        if not nk_values:
            raise ValueError("No valid image numbers provided.")
        
        ks = self[inums[0]]["k"]  # Assuming all inums have the same k values
        means = np.mean(nk_values, axis=0)
        errs = np.std(nk_values, axis=0) / np.sqrt(len(nk_values)) if len(nk_values) > 1 else np.zeros_like(means)

        return ks, means, errs
    
    def get_averaged_calc(self, inums: list[int], calcparam="N") -> tuple[float, float]:
        inums = self._remove_blanks(inums)
        calc_values = []
        for inum in inums:
            try:
                calc_values.append(self[inum]["calc"][calcparam])
            except KeyError:
                print(f"Warning: No data for image number {inum}, skipping.")
        
        if not calc_values:
            raise ValueError("No valid image numbers provided.")
        
        mean = np.mean(calc_values)
        err = np.std(calc_values) / np.sqrt(len(calc_values)) if len(calc_values) > 1 else 0

        return mean, err
    
    def plot_over_individual_runs(self,inums: list[int], calcparam="N") -> list[float]:
        inums = self._remove_blanks(inums)
        calc_values = []
        for inum in inums:
            try:
                calc_values.append(self[inum]["calc"][calcparam])
            except KeyError:
                print(f"Warning: No data for image number {inum}, skipping.")
        return calc_values

    
    def plottable_nk(self,inums_lists: list[list[int]]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        init_set_data = [self.get_averaged_nk(inums) for inums in inums_lists]
        ks_lists = [data[0] for data in init_set_data]
        nks_lists = [data[1] for data in init_set_data]
        errs_lists = [data[2] for data in init_set_data]
        return ks_lists, nks_lists, errs_lists
    
    def plottable_calc(self, inums_lists: list[list[int]], calcparam="N") -> tuple[list[float], list[float]]:
        means_errs = [self.get_averaged_calc(inums, calcparam) for inums in inums_lists]
        means = [me[0] for me in means_errs]
        errs = [me[1] for me in means_errs]
        return means, errs
        

    # ---------- private methods ----------

    def _load_data(self) -> None:
        filepath_k = f"{self.directory}/k3d_{self.suffix}.txt"
        filepath_nk = f"{self.directory}/nk3d_{self.suffix}.txt"
        if self._twoD:
            filepath_k = f"{self.directory}/k2d_{self.suffix}.txt"
            filepath_nk = f"{self.directory}/nk2d_{self.suffix}.txt"
        filepath_ds = f"{self.directory}/ds_{self.suffix}.txt"

        # k3d/nk3d files are large CSV-like files where the first line is the
        # column names (e.g. i591,i592,...) and subsequent lines contain
        # one k (or nk) value per column. numpy.genfromtxt struggles on some
        # of these files (mixed types/encodings), so parse them robustly here.
        def _read_matrix_file(path: str) -> Dict[str, np.ndarray]:
            with open(path, "r", encoding="utf-8", errors="surrogateescape") as f:
                header = f.readline().strip()
                if not header:
                    raise ValueError(f"Empty header in {path}")
                fields = header.split(",")
                ncols = len(fields)
                cols: List[List[float]] = [[] for _ in range(ncols)]

                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != ncols:
                        # If line length mismatch, try to pad/truncate to ncols
                        if len(parts) < ncols:
                            parts += [""] * (ncols - len(parts))
                        else:
                            parts = parts[:ncols]
                    for i, p in enumerate(parts):
                        try:
                            cols[i].append(float(p))
                        except Exception:
                            # treat empty as nan
                            cols[i].append(float("nan"))

                return {name: np.asarray(col, dtype=float) for name, col in zip(fields, cols)}

        self.k_data = _read_matrix_file(filepath_k)
        self.nk_data = _read_matrix_file(filepath_nk)
        # For the ds file, detect delimiter (tab or comma) then load
        with open(filepath_ds, 'r', encoding='utf-8', errors='surrogateescape') as f:
            first = f.readline()
        ds_delim = '\t' if '\t' in first else ','
        self.calc_data = np.genfromtxt(filepath_ds, names=True, delimiter=ds_delim)


    def _get_from_inum(self, inum: int) -> Dict[str, Any]:
        field = f"i{inum}"

        try:
            k_values = self.k_data[field]
            nk_values = self.nk_data[field]
        except ValueError as e:
            raise KeyError(f"No k/nk data for image number {inum}") from e

        # Some k/nk files are read by genfromtxt as strings containing comma-separated
        # lists (or as arrays of byte-strings). Normalize into numpy float arrays here.
        def _parse_value(val):
            # Handle numpy scalars
            if isinstance(val, (bytes, str)):
                s = val.decode() if isinstance(val, bytes) else val
                parts = [p for p in s.split(',') if p.strip() != '']
                return np.asarray([float(p) for p in parts], dtype=float)
            if isinstance(val, np.ndarray):
                # If numeric already, return as float array
                if np.issubdtype(val.dtype, np.number):
                    return np.asarray(val, dtype=float)
                # If 0-d array with a string-like item
                if val.ndim == 0:
                    item = val.item()
                    return _parse_value(item)
                # 1-d array of strings or bytes
                if val.dtype.kind in ('U', 'S'):
                    try:
                        return val.astype(float)
                    except Exception:
                        # join all entries and split by comma
                        joined = ','.join([x.decode() if isinstance(x, bytes) else str(x) for x in val])
                        parts = [p for p in joined.split(',') if p.strip() != '']
                        return np.asarray([float(p) for p in parts], dtype=float)
                # object array: maybe contains lists
                if val.dtype == object:
                    first = val.flat[0]
                    if isinstance(first, (list, tuple, np.ndarray)):
                        return np.asarray(first, dtype=float)
                    return _parse_value(first)
            # Fallback: try converting directly
            try:
                return np.asarray(val, dtype=float)
            except Exception:
                raise ValueError(f"Cannot parse k/nk values for image {inum}: {val}")

        k_arr = _parse_value(k_values)
        nk_arr = _parse_value(nk_values)

        # Locate the row in calc_data whose ImageNumber matches inum.
        # Use integer comparison to avoid issues if genfromtxt produced floats.
        image_numbers = np.array(self.calc_data["ImageNumber"], dtype=int)
        matches = np.nonzero(image_numbers == inum)[0]
        if matches.size == 0:
            raise IndexError(f"Image number {inum} out of range")

        idx = int(matches[0])
        calc_values = self.calc_data[idx]

        return {
            "k": k_arr,
            "nk": nk_arr,
            "calc": calc_values,
        }
    
    def _remove_blanks(self, inums: list[int]) -> list[int]:
        return [inum for inum in inums if inum not in self._blanks]