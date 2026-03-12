"""
E3SM data processing script with config-driven pipeline.
Handles files that span month boundaries (mid-month to mid-next-month).
"""

import dataclasses
import glob
import logging
import os
import sys
from typing import List, Optional, Sequence, Tuple, Dict

import click
import dacite
import numpy as np
import xarray as xr
import yaml
from collections import defaultdict

# Constants
LATENT_HEAT_OF_VAPORIZATION = 2.501e6  # J/kg
GRAVITY = 9.80665  # m/s^2
REFERENCE_PRESSURE = 1e5  # Pa

# Default variables to extract after merging (can be overridden in config)
DEFAULT_E3SM_VARIABLES = [
    'PS', 'TS', 'T', 'U', 'V', 'Q', 'OMEGA', 'CLDLIQ', 'CLDICE',
    'RAINQM', 'TMQ', 'TGCLDLWP', 'TGCLDIWP', 'OCNFRAC', 'LANDFRAC',
    'ICEFRAC', 'QREFHT', 'TREFHT', 'U10', 'PRECT', 'LHFLX', 'SHFLX',
    'FLNS', 'FLDS', 'FSNS', 'FSDS', 'FSNTOA', 'SOLIN', 'FLUT',
    'PRECSC', 'PRECSL', 'QFLX', 'hyai', 'hybi'
]


@dataclasses.dataclass
class DataPaths:
    """Paths to input data files with glob patterns.

    Note: h1_pattern is now optional because all data may be in h0 files.
    """
    h1_pattern: Optional[str] = None
    h2_pattern: Optional[str] = None
    h0_pattern: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "DataPaths":
        return dacite.from_dict(data_class=cls, data=data)


@dataclasses.dataclass
class VerticalCoarseningConfig:
    """Configuration for vertical coarsening."""

    indices: Sequence[Tuple[int, int]]
    variables: Sequence[str] = dataclasses.field(
        default_factory=lambda: [
            "T", "specific_total_water", "U", "V", "OMEGA"
        ]
    )
    validate: bool = True


@dataclasses.dataclass
class ProcessingConfig:
    """Main processing configuration."""

    data_paths: DataPaths
    output_directory: str
    vertical_coarsening: VerticalCoarseningConfig
    start_year: int = 1950
    end_year: int = 1950
    start_month: int = 1
    end_month: int = 12
    e3sm_variables: Optional[List[str]] = None

    @classmethod
    def from_file(cls, path: str) -> "ProcessingConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Handle nested dataclass creation
        if "data_paths" in data:
            data["data_paths"] = DataPaths.from_dict(data["data_paths"])
        if "vertical_coarsening" in data:
            vc_data = data["vertical_coarsening"]
            data["vertical_coarsening"] = VerticalCoarseningConfig(
                indices=[tuple(idx) for idx in vc_data.get("indices", [])],
                variables=vc_data.get(
                    "variables",
                    ["T", "specific_total_water", "U", "V", "OMEGA"]
                ),
                validate=vc_data.get("validate", True),
            )

        return dacite.from_dict(
            data_class=cls,
            data=data,
            config=dacite.Config(cast=[tuple], strict=False),
        )
    
    def get_e3sm_variables(self) -> List[str]:
        """Get E3SM variables list (use config or default)."""
        return self.e3sm_variables if self.e3sm_variables is not None else DEFAULT_E3SM_VARIABLES


def group_files_by_month(
    file_pattern: str,
    year: int,
    file_type: str = "unknown",
    static_fields_only: bool = False,
) -> Dict[int, List[str]]:
    """
    Group files by months they contain data for.

    Since files span month boundaries (mid-month to mid-next-month),
    each file will be associated with 2 months.

    Args:
        file_pattern: Glob pattern with {year} placeholder
        year: Year to process
        file_type: Description for logging
        static_fields_only: If True, just find any file from the year (for h0 static fields)

    Returns:
        Dictionary mapping month number (1-12) to list of file paths
    """
    import cftime
    import pandas as pd
    import re

    year_str = str(year)

    # Get files for this year and potentially previous/next year
    # (files may span year boundaries)
    patterns_to_search = [
        file_pattern.replace("{year}", str(year - 1)),
        file_pattern.replace("{year}", year_str),
        file_pattern.replace("{year}", str(year + 1)),
    ]

    all_files = []
    for pattern in patterns_to_search:
        pattern_expanded = pattern.replace("{month}", "*")
        files = glob.glob(pattern_expanded)
        all_files.extend(files)

    all_files = sorted(set(all_files))  # Remove duplicates and sort

    if not all_files:
        raise FileNotFoundError(
            f"No {file_type} files found for year {year}"
        )

    logging.info(f"Found {len(all_files)} total {file_type} file(s) around year {year}")

    # For static fields (h0), just assign the first file to all months
    if static_fields_only:
        logging.info(f"Using simple mode for {file_type} (static fields only)")
        # Find any file from the target year
        year_files = [f for f in all_files if f"/{year}-" in f or f".{year}-" in f]
        if not year_files:
            year_files = all_files[:1]  # Fallback to first file

        # Assign the same file(s) to all months
        month_to_files = {month: year_files[:1] for month in range(1, 13)}

        filename = os.path.basename(year_files[0])
        logging.info(f"  Using file for all months: {filename}\n")

        return month_to_files

    # First: Group files by month based on their actual time coordinates
    month_to_files = defaultdict(list)

    for filepath in all_files:
        try:
            # Open file and read time coordinate
            with xr.open_dataset(filepath, decode_times=True) as ds:
                if 'time' not in ds.coords:
                    logging.warning(f"No time coordinate in {filepath}, skipping")
                    continue

                # Get all year-months present in this file
                times = ds.time.values
                year_months_in_file = set()

                # Convert each time value to year-month tuple
                for time_val in times:
                    # Handle both cftime and numpy datetime64
                    if hasattr(time_val, 'year') and hasattr(time_val, 'month'):
                        # cftime object or similar - has year/month attributes directly
                        file_year = time_val.year
                        file_month = time_val.month
                    else:
                        # numpy datetime64 - convert to pandas Timestamp
                        ts = pd.Timestamp(time_val)
                        file_year = ts.year
                        file_month = ts.month

                    year_months_in_file.add((file_year, file_month))

                # Add this file to all months it contains FOR THE TARGET YEAR
                for file_year, month in year_months_in_file:
                    if file_year == year:
                        month_to_files[month].append(filepath)

        except Exception as e:
            logging.warning(f"Error reading {filepath}: {e}, skipping")
            continue

    # If file_type is h0: prefer filename matches for each month when available.
    if file_type.lower() == "h0":
        # regex to find "YYYY-MM" in basename
        for month in range(1, 13):
            pattern = re.compile(rf"{re.escape(year_str)}-{month:02d}")
            matched = [f for f in all_files if pattern.search(os.path.basename(f))]
            if matched:
                # Prefer filename matches: replace whatever time-based mapping had
                month_to_files[month] = sorted(set(matched))
                logging.info(
                    f"h0 filename preference: assigned {len(matched)} file(s) to {year}-{month:02d}"
                )

    # Log the grouping results with file details
    logging.info(f"\n{file_type.upper()} Files grouped by month for year {year}:")
    for month in sorted(month_to_files.keys()):
        logging.info(f"  {year}-{month:02d}: {len(month_to_files[month])} file(s)")
        for filepath in month_to_files[month]:
            # Extract just the filename for cleaner logging
            filename = os.path.basename(filepath)
            logging.info(f"    - {filename}")

    return dict(month_to_files)   

def open_files_for_month(
    file_list: List[str],
    year: int,
    month: int,
    file_type: str = "unknown",
) -> xr.Dataset:
    """
    Open files and filter to specific year-month.
    
    Args:
        file_list: List of file paths to open
        year: Year to filter
        month: Month to filter (1-12)
        file_type: Description for logging
    
    Returns:
        xarray Dataset filtered to the specified year-month
    """
    logging.info(
        f"\nOpening {len(file_list)} {file_type} file(s) for {year}-{month:02d}:"
    )
    
    if not file_list:
        raise ValueError(f"No files provided for {year}-{month:02d}")
    
    # Log each file being opened
    for filepath in file_list:
        filename = os.path.basename(filepath)
        logging.info(f"  - {filename}")
    
    # Open all files - let xarray auto-detect how to combine
    if len(file_list) == 1:
        # Single file - just open it
        ds = xr.open_dataset(file_list[0])
    else:
        # Multiple files - combine by coordinates
        ds = xr.open_mfdataset(
            file_list, 
            combine='by_coords',
            data_vars='minimal',
            coords='minimal',
            compat='override',
        )
    
    # Filter to the specific year-month
    time_slice = f"{year}-{month:02d}"

    if file_type!= 'h0':
        ds_month = ds.sel(time=time_slice)
    else:
        ds_month = ds
    
    if len(ds_month.time) == 0:
        raise ValueError(
            f"No data found for {year}-{month:02d} after filtering. "
            f"Files: {file_list}"
        )
    
    return ds_month
    

def select_variables(
    ds: xr.Dataset,
    var_list: Sequence[str],
    dataset_name: str = "dataset",
) -> xr.Dataset:
    """Select only specified variables from dataset."""
    available_vars = [v for v in var_list if v in ds.data_vars or v in ds.coords]
    missing_vars = set(var_list) - set(available_vars)

    if missing_vars:
        logging.debug(
            f"Variables not found in {dataset_name}: {missing_vars}"
        )

    if available_vars:
        logging.info(f"Selected {len(available_vars)} variables from {dataset_name}")
        return ds[available_vars]
    else:
        logging.warning(f"No variables selected from {dataset_name}")
        return xr.Dataset()


def compute_pressure_thickness_mid(
    ds: xr.Dataset,
    lev_dim: str = "lev",
    ilev_dim: str = "ilev",
    hyai: str = "hyai",
    hybi: str = "hybi",
    ps: str = "PS",
    p0: float = REFERENCE_PRESSURE,
    out_var: str = "dp",
) -> xr.Dataset:
    """Compute pressure thickness from hybrid interface levels."""
    ps_broadcasted = ds[ps].expand_dims({ilev_dim: ds[ilev_dim]}, axis=3)
    pmid = ds[hyai] * p0 + ds[hybi] * ps_broadcasted

    dp = (
        pmid.diff(dim=ilev_dim)
        .rename({ilev_dim: lev_dim})
        .assign_coords({lev_dim: (lev_dim, ds[lev_dim].values)})
    )

    dp.name = out_var
    dp.attrs["units"] = "Pa"
    dp.attrs[
        "long_name"
    ] = "pressure thickness (computed from interface-levels)"

    return ds.assign({out_var: dp})


def compute_specific_total_water_e3sm(
    ds: xr.Dataset,
    output_name: str = "specific_total_water",
) -> xr.Dataset:
    """Compute specific total water from E3SM water species."""
    e3sm_species = {
        "Q": "Water vapor",
        "CLDLIQ": "Cloud liquid water",
        "CLDICE": "Cloud ice",
        "RAINQM": "Rain water",
        "SNOWQM": "Snow water",
    }

    available_species = {}
    for var_name, description in e3sm_species.items():
        if var_name in ds.data_vars:
            available_species[var_name] = description

    if not available_species:
        raise ValueError(
            f"No E3SM water species found. Expected one or more of: "
            f"{list(e3sm_species.keys())}"
        )

    logging.info(
        f"Computing {output_name} from {len(available_species)} "
        f"water species"
    )

    specific_total_water = sum(
        [ds[name] for name in available_species.keys()]
    )

    species_str = " + ".join(available_species.keys())
    specific_total_water.attrs["units"] = "kg/kg"
    specific_total_water.attrs[
        "long_name"
    ] = f"Specific total water ({species_str})"
    specific_total_water.attrs["standard_name"] = "specific_total_water"
    specific_total_water.attrs["formula"] = species_str

    return ds.assign({output_name: specific_total_water})


def compute_surface_precipitation_rate(
    ds: xr.Dataset,
    total_precip_rate_name: str = "PRECT",
    liquid_precip_density: float = 1e3,
    output_name: str = "surface_precipitation_rate",
) -> xr.Dataset:
    """Compute surface precipitation rate for E3SM data."""
    if total_precip_rate_name not in ds.data_vars:
        raise ValueError(
            f"Variable '{total_precip_rate_name}' not found in dataset"
        )

    precip = ds[total_precip_rate_name]
    precip_mass_flux = precip * liquid_precip_density

    precip_mass_flux.attrs["units"] = "kg/m2/s"
    precip_mass_flux.attrs["long_name"] = "Surface precipitation rate"
    precip_mass_flux.attrs["standard_name"] = "precipitation_flux"

    return ds.assign({output_name: precip_mass_flux})


def compute_rad_fluxes(ds: xr.Dataset) -> xr.Dataset:
    """Compute radiative flux variables for E3SM data."""
    rad_flux_formulas = {
        "surface_upward_longwave_flux": (
            lambda x, y: x + y,
            "FLNS",
            "FLDS",
        ),
        "surface_upward_shortwave_flux": (
            lambda x, y: x - y,
            "FSDS",
            "FSNS",
        ),
        "top_of_atmos_upward_shortwave_flux": (
            lambda x, y: x - y,
            "SOLIN",
            "FSNTOA",
        ),
    }

    fluxes = {}

    for output_name, (formula, var1, var2) in rad_flux_formulas.items():
        if var1 not in ds.data_vars or var2 not in ds.data_vars:
            logging.warning(
                f"Skipping {output_name}: missing {var1} or {var2}"
            )
            continue

        result = formula(ds[var1], ds[var2])
        result.attrs["units"] = "W/m2"
        result.attrs["long_name"] = output_name.replace("_", " ")
        result.attrs["standard_name"] = output_name

        fluxes[output_name] = result

    return ds.assign(fluxes)


def compute_total_water_path(
    ds: xr.Dataset,
    specific_total_water_name: str,
    pressure_thickness_name: str,
    output_name: str = "total_water_path",
) -> xr.Dataset:
    """Compute total water path by vertical integration."""
    twp = (
        ds[specific_total_water_name] * ds[pressure_thickness_name]
    ).sum(dim="lev") / GRAVITY
    twp.attrs["units"] = "kg/m2"
    twp.attrs["long_name"] = "Total water path"
    return ds.assign({output_name: twp})


def compute_twp_tendency(
    ds: xr.Dataset,
    total_water_path_name: str = "total_water_path",
    time_dim: str = "time",
    output_name: str = "tendency_of_total_water_path",
) -> xr.Dataset:
    """Compute time tendency of total water path."""
    twp = ds[total_water_path_name]
    dt = ds[time_dim].diff(time_dim)

    tendency = twp.diff(time_dim) / dt.astype(
        "timedelta64[s]"
    ).astype(float)

    # Forward fill first timestep NaN
    tendency = tendency.fillna(tendency.isel({time_dim: 0}))

    tendency.attrs["units"] = "kg/m2/s"
    tendency.attrs["long_name"] = "Tendency of total water path"

    return ds.assign({output_name: tendency})


def compute_twp_advective_tendency(
    ds: xr.Dataset,
    twp_tendency_name: str = "tendency_of_total_water_path",
    latent_heat_flux_name: str = "LHFLX",
    precip_name: str = "surface_precipitation_rate",
    latent_heat_of_vaporization: float = LATENT_HEAT_OF_VAPORIZATION,
    output_name: str = (
        "tendency_of_total_water_path_due_to_advection"
    ),
) -> xr.Dataset:
    """Compute advective moisture tendency from moisture budget."""
    evaporation = ds[latent_heat_flux_name] / latent_heat_of_vaporization
    precipitation = ds[precip_name]

    advection = (
        ds[twp_tendency_name] - evaporation + precipitation
    )

    # Forward fill first timestep NaN
    advection = advection.fillna(advection.isel(time=0))

    advection.attrs["units"] = "kg/m2/s"
    advection.attrs[
        "long_name"
    ] = "Tendency of total water path due to advection"

    return ds.assign({output_name: advection})


def validate_vertical_coarsening_indices(
    dim_size: int,
    interface_indices: Sequence[Tuple[int, int]],
    vertical_type: str = "vertical",
    context: str = "",
) -> None:
    """Validate vertical coarsening indices."""
    for i, (start, end) in enumerate(interface_indices):
        if not (0 <= start < end <= dim_size):
            raise ValueError(
                f"Invalid interface_indices[{i}] = ({start}, {end}) in "
                f"{vertical_type} dimension (size = {dim_size}) from "
                f"context '{context}'."
            )


def compute_vertical_coarsening(
    ds: xr.Dataset,
    vertically_resolved_names: Sequence[str],
    interface_indices: Sequence[Tuple[int, int]],
    dim: str,
    pressure_thickness_name: str,
    validate_indices: bool = True,
) -> Tuple[xr.Dataset, dict]:
    """Compute vertical coarsening by mass-weighted mean."""
    if validate_indices:
        validate_vertical_coarsening_indices(
            ds.sizes[dim],
            interface_indices,
            vertical_type="atmosphere",
            context="compute_vertical_coarsening",
        )

    coarsened_arrays = {}

    for i, (start, end) in enumerate(interface_indices):
        dp_slice = ds[pressure_thickness_name].isel(
            {dim: slice(start, end)}
        )

        for name in vertically_resolved_names:
            if name not in ds:
                continue

            var_slice = ds[name].isel({dim: slice(start, end)})
            coarsened = (
                (var_slice * dp_slice).sum(dim) / dp_slice.sum(dim)
            )
            coarsened.attrs["long_name"] = (
                f"{ds[name].attrs.get('long_name', name)} level-{i}"
            )
            coarsened_arrays[f"{name}_{i}"] = coarsened

    ds = ds.drop_vars(vertically_resolved_names, errors="ignore")

    return ds.assign(coarsened_arrays), coarsened_arrays


def compute_coarse_ak_bk(
    ds: xr.Dataset,
    interface_indices: Sequence[Tuple[int, int]],
    z_dim: str,
    hybrid_level_coeffs: List[str],
    reference_pressure: float = REFERENCE_PRESSURE,
) -> xr.Dataset:
    """Compute coarse ak and bk coordinates for vertical interfaces."""
    data = {}
    hyai, hybi = hybrid_level_coeffs

    for i, (start, end) in enumerate(interface_indices):
        data[f"ak_{i}"] = (
            ds[hyai].isel({z_dim: start}) * reference_pressure
        )
        data[f"bk_{i}"] = ds[hybi].isel({z_dim: start})
        if i == len(interface_indices) - 1:
            data[f"ak_{i + 1}"] = (
                ds[hyai].isel({z_dim: end}) * reference_pressure
            )
            data[f"bk_{i + 1}"] = ds[hybi].isel({z_dim: end})

    for i in range(len(interface_indices) + 1):
        data[f"ak_{i}"].attrs["units"] = "Pa"
        data[f"bk_{i}"].attrs["units"] = ""
        for name in ["ak", "bk"]:
            data[f"{name}_{i}"] = data[f"{name}_{i}"].drop_vars(z_dim, errors="ignore")

    return xr.merge([ds, xr.Dataset(data)])


def get_coarse_variable_names(
    ds: xr.Dataset,
    base_name: str,
) -> List[str]:
    """Get all coarsened variables for a given base name."""
    coarse_vars = [
        var for var in ds.data_vars
        if var.startswith(f"{base_name}_") and
        var[len(base_name) + 1:].isdigit()
    ]
    return sorted(coarse_vars, key=lambda x: int(x.split("_")[-1]))


def get_ak_bk_variable_names(ds: xr.Dataset) -> Tuple[List[str], List[str]]:
    """Get all ak and bk variable names from dataset."""
    ak_vars = sorted(
        [v for v in ds.data_vars if v.startswith("ak_")],
        key=lambda x: int(x.split("_")[-1])
    )
    bk_vars = sorted(
        [v for v in ds.data_vars if v.startswith("bk_")],
        key=lambda x: int(x.split("_")[-1])
    )
    return ak_vars, bk_vars


def create_forcing_dataset(
    ds: xr.Dataset,
) -> xr.Dataset:
    """Create forcing dataset with required variables."""
    logging.info("Creating forcing dataset...")

    base_forcing_vars = [
        'ICEFRAC', 'LANDFRAC', 'OCNFRAC', 'PHIS', 'SOLIN', 'TS'
    ]

    forcing_vars = [v for v in base_forcing_vars if v in ds.data_vars]
    ak_vars, bk_vars = get_ak_bk_variable_names(ds)
    forcing_vars.extend(ak_vars)
    forcing_vars.extend(bk_vars)

    forcing_ds = ds[forcing_vars]

    logging.info(f"Forcing dataset created with {len(forcing_ds.data_vars)} "
                 f"variables")

    return forcing_ds


def create_initial_condition_dataset(
    ds: xr.Dataset,
) -> xr.Dataset:
    """Create initial condition dataset with required variables."""
    logging.info("Creating initial condition dataset...")

    base_ic_vars = [
        'FLDS', 'FLUT', 'FSDS', 'ICEFRAC', 'LANDFRAC', 'LHFLX',
        'OCNFRAC', 'PHIS', 'PS', 'SHFLX', 'SOLIN', 'TS',
        'surface_precipitation_rate', 'surface_upward_longwave_flux',
        'surface_upward_shortwave_flux', 'tendency_of_total_water_path',
        'tendency_of_total_water_path_due_to_advection',
        'top_of_atmos_upward_shortwave_flux', 'total_water_path'
    ]

    ic_vars = [v for v in base_ic_vars if v in ds.data_vars]

    for base_var in ['T', 'U', 'V', 'specific_total_water']:
        coarse_vars = get_coarse_variable_names(ds, base_var)
        ic_vars.extend(coarse_vars)

    ak_vars, bk_vars = get_ak_bk_variable_names(ds)
    ic_vars.extend(ak_vars)
    ic_vars.extend(bk_vars)

    ic_ds = ds[ic_vars]

    logging.info(f"Initial condition dataset created with "
                 f"{len(ic_ds.data_vars)} variables")

    return ic_ds

def convert_to_float32(ds: xr.Dataset) -> xr.Dataset:
    """Convert all variables and coordinates to float32."""
    logging.info("Converting all variables to float32...")
    
    # Convert all data variables
    for var in ds.data_vars:
        if ds[var].dtype != np.float32:
            ds[var] = ds[var].astype(np.float32)
    
    # Convert coordinates (except time)
    for coord in ds.coords:
        if coord != 'time' and ds[coord].dtype != np.float32:
            if np.issubdtype(ds[coord].dtype, np.floating):
                ds[coord] = ds[coord].astype(np.float32)
    
    logging.info("All variables converted to float32")
    return ds


def process_e3sm_month(
    config: ProcessingConfig,
    year: int,
    month: int,
    h1_files_by_month: Optional[Dict[int, List[str]]] = None,
    h2_files_by_month: Optional[Dict[int, List[str]]] = None,
    h0_files_by_month: Optional[Dict[int, List[str]]] = None,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Process E3SM data for a single month.

    Behavior change: h1 and h2 inputs are optional. If no h1/h2 files are
    available for the month, the function will attempt to use h0 files as the
    primary source (all data now in h0).

    Returns:
        Tuple of (full_dataset, forcing_dataset, initial_condition_dataset)
    """
    month_str = f"{month:02d}"
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing year-month: {year}-{month_str}")
    logging.info(f"{'='*60}")

    # Get variable list from config
    e3sm_variables = config.get_e3sm_variables()
    logging.info(f"Using {len(e3sm_variables)} E3SM variables")

    # Prepare input dataset (prefer h1/h2 if present; otherwise use h0)
    ds_primary = None

    # Try h1/h2 first if files are provided and present for this month
    h1_available = h1_files_by_month and (month in h1_files_by_month) and h1_files_by_month.get(month)
    if h1_available:
        ds_h1 = open_files_for_month(
            h1_files_by_month[month],
            year,
            month,
            file_type="h1",
        )
        logging.info("Selecting variables from h1 dataset...")
        ds_h1_selected = select_variables(ds_h1, e3sm_variables, "h1")
        ds_primary = ds_h1_selected

    # If h2 present, open and merge (only if h1 present or h2 has data)
    if h2_files_by_month and (month in h2_files_by_month) and h2_files_by_month.get(month):
        ds_h2 = open_files_for_month(
            h2_files_by_month[month],
            year,
            month,
            file_type="h2",
        )
        logging.info("Selecting variables from h2 dataset...")
        ds_h2_selected = select_variables(ds_h2, e3sm_variables, "h2")
        if ds_primary is not None and len(ds_h2_selected.data_vars) > 0:
            logging.info("Merging h1 and h2 datasets...")
            ds_primary = ds_primary.merge(ds_h2_selected)
        elif ds_primary is None:
            ds_primary = ds_h2_selected

    # If no h1/h2 primary dataset, try h0 (all data now in h0)
    if ds_primary is None:
        if not h0_files_by_month or month not in h0_files_by_month or not h0_files_by_month.get(month):
            raise ValueError(
                f"No input files found for {year}-{month:02d}: "
                f"no h1/h2 and no h0 files available for this month"
            )
        ds_h0 = open_files_for_month(
            h0_files_by_month[month],
            year,
            month,
            file_type="h0",
        )
        logging.info("Selecting variables from h0 dataset (primary)...")
        ds_primary = select_variables(ds_h0, e3sm_variables, "h0")
        # If PHIS or other static-like fields are present in h0, they are
        # already in ds_primary; no separate h0 extraction is needed.
        try:
            ds_h0.close()
        except Exception:
            pass

    ds_p = ds_primary

    logging.info(f"Merged dataset shape: {ds_p.dims}")

    # Load into memory
    logging.info("Loading dataset into memory...")
    ds_p = ds_p.load()

    # If an h0 file exists and contains PHIS and h1/h2 didn't supply it,
    # try to add PHIS from h0 (preserve original behavior)
    if (('PHIS' not in ds_p.data_vars) and h0_files_by_month and (month in h0_files_by_month) and h0_files_by_month.get(month)):
        try:
            ds_h0 = open_files_for_month(
                h0_files_by_month[month],
                year,
                month,
                file_type="h0",
            )
            if "PHIS" in ds_h0.data_vars:
                logging.info("Adding PHIS from h0 dataset...")
                ds_p["PHIS"] = ds_h0.isel(time=0).PHIS.load()
            ds_h0.close()
        except Exception as e:
            logging.warning(f"Could not extract PHIS from h0 for {year}-{month:02d}: {e}")

    # Compute derived variables
    logging.info("Computing pressure thickness...")
    ds_p = compute_pressure_thickness_mid(ds_p)

    logging.info("Computing specific total water...")
    ds_p = compute_specific_total_water_e3sm(ds_p)

    logging.info("Computing radiative fluxes...")
    ds_p = compute_rad_fluxes(ds_p)

    logging.info("Computing surface precipitation rate...")
    ds_p = compute_surface_precipitation_rate(ds_p)

    logging.info("Computing total water path and tendencies...")
    ds_p = compute_total_water_path(
        ds_p,
        "specific_total_water",
        "dp",
    )
    ds_p = compute_twp_tendency(ds_p)
    ds_p = compute_twp_advective_tendency(ds_p)

    # Vertical coarsening
    logging.info("Performing vertical coarsening...")
    ds_p, _ = compute_vertical_coarsening(
        ds_p,
        vertically_resolved_names=config.vertical_coarsening.variables,
        interface_indices=config.vertical_coarsening.indices,
        dim="lev",
        pressure_thickness_name="dp",
        validate_indices=config.vertical_coarsening.validate,
    )

    logging.info("Computing coarse ak/bk coefficients...")
    ds_p = compute_coarse_ak_bk(
        ds_p,
        config.vertical_coarsening.indices,
        "ilev",
        ["hyai", "hybi"],
        REFERENCE_PRESSURE,
    )

    # Sort variables
    ds_p = ds_p[sorted(ds_p.data_vars)]

    # Convert ALL variables to float32 before output
    logging.info("Converting all variables to float32...")
    ds_p = convert_to_float32(ds_p)
    
    # Create forcing and initial condition datasets
    forcing_ds = create_forcing_dataset(ds_p)
    ic_ds = create_initial_condition_dataset(ds_p)

    return ds_p, forcing_ds, ic_ds


@click.command()
@click.option(
    "--config",
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--start-year",
    default=None,
    type=int,
    help="Start year to process (overrides config)",
)
@click.option(
    "--end-year",
    default=None,
    type=int,
    help="End year to process (overrides config)",
)
@click.option(
    "--start-month",
    default=None,
    type=int,
    help="Start month to process (1-12, overrides config)",
)
@click.option(
    "--end-month",
    default=None,
    type=int,
    help="End month to process (1-12, overrides config)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Print debug information",
)
def main(config: str, start_year: int, end_year: int, start_month: int, end_month: int, debug: bool):
    """Process E3SM data according to configuration."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        # Load configuration
        logging.info(f"Loading configuration from {config}")
        proc_config = ProcessingConfig.from_file(config)

        # Override parameters from command line if provided
        if start_year is not None:
            proc_config.start_year = start_year
        if end_year is not None:
            proc_config.end_year = end_year
        if start_month is not None:
            proc_config.start_month = start_month
        if end_month is not None:
            proc_config.end_month = end_month

        # Validate months
        if not (1 <= proc_config.start_month <= 12):
            raise ValueError(f"start_month must be between 1 and 12, got {proc_config.start_month}")
        if not (1 <= proc_config.end_month <= 12):
            raise ValueError(f"end_month must be between 1 and 12, got {proc_config.end_month}")

        # Create output directory
        os.makedirs(proc_config.output_directory, exist_ok=True)

        logging.info(f"\n{'='*60}")
        logging.info(f"Starting E3SM data processing")
        logging.info(f"Years: {proc_config.start_year} to {proc_config.end_year}")
        logging.info(f"Months: {proc_config.start_month} to {proc_config.end_month}")
        logging.info(f"Output directory: {proc_config.output_directory}")
        logging.info(f"{'='*60}\n")

        # Process each year
        for year in range(proc_config.start_year, proc_config.end_year + 1):
            logging.info(f"\n{'#'*60}")
            logging.info(f"Processing year {year}")
            logging.info(f"{'#'*60}\n")
            
            # Group files by month for this year
            # (this will also search prev/next year for files that span boundaries)
            h1_files_by_month = {}
            if proc_config.data_paths.h1_pattern:
                try:
                    h1_files_by_month = group_files_by_month(
                        proc_config.data_paths.h1_pattern,
                        year,
                        file_type="h1",
                    )
                except FileNotFoundError as e:
                    logging.warning(f"No h1 files found for year {year}: {e}")
                    h1_files_by_month = {}

            h2_files_by_month = {}
            if proc_config.data_paths.h2_pattern:
                try:
                    h2_files_by_month = group_files_by_month(
                        proc_config.data_paths.h2_pattern,
                        year,
                        file_type="h2",
                    )
                except FileNotFoundError as e:
                    logging.warning(f"No h2 files found for year {year}: {e}")
                    h2_files_by_month = {}

            h0_files_by_month = {}
            if proc_config.data_paths.h0_pattern:
                try:
                    h0_files_by_month = group_files_by_month(
                        proc_config.data_paths.h0_pattern,
                        year,
                        file_type="h0",
                    )
                except FileNotFoundError as e:
                    logging.warning(f"No h0 files found for year {year}: {e}")
                    h0_files_by_month = {}
            
            # Determine month range for this year
            if year == proc_config.start_year:
                month_start = proc_config.start_month
            else:
                month_start = 1
                
            if year == proc_config.end_year:
                month_end = proc_config.end_month
            else:
                month_end = 12
            
            # Process each month
            for month in range(month_start, month_end + 1):
                try:
                    # Process dataset for this month
                    ds_full, ds_forcing, ds_ic = process_e3sm_month(
                        proc_config,
                        year,
                        month,
                        h1_files_by_month,
                        h2_files_by_month,
                        h0_files_by_month,
                    )

                    # Define output paths
                    month_str = f"{month:02d}"
                    full_output_path = os.path.join(
                        proc_config.output_directory,
                        f"{year}-{month_str}.nc",
                    )
                    forcing_output_path = os.path.join(
                        proc_config.output_directory,
                        f"{year}-{month_str}.forcing.nc",
                    )
                    ic_output_path = os.path.join(
                        proc_config.output_directory,
                        f"{year}-{month_str}.ic.nc",
                    )

                    # Save outputs
                    logging.info(f"Saving full dataset to {full_output_path}")
                    ds_full.to_netcdf(full_output_path)

                    logging.info(f"Saving forcing dataset to {forcing_output_path}")
                    ds_forcing.to_netcdf(forcing_output_path)

                    logging.info(f"Saving initial condition dataset to {ic_output_path}")
                    ds_ic.to_netcdf(ic_output_path)

                    logging.info(f"\n✓ Year-Month {year}-{month_str} completed successfully!")
                    logging.info(f"  Full: {full_output_path}")
                    logging.info(f"  Forcing: {forcing_output_path}")
                    logging.info(f"  Initial Condition: {ic_output_path}\n")

                    # Clean up to free memory
                    ds_full.close()
                    ds_forcing.close()
                    ds_ic.close()
                    del ds_full, ds_forcing, ds_ic

                except Exception as e:
                    logging.error(f"Error processing {year}-{month:02d}: {e}", exc_info=True)
                    logging.info(f"Continuing to next month...\n")
                    continue

        logging.info(f"\n{'='*60}")
        logging.info(f"All months processed!")
        logging.info(f"{'='*60}")

    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()