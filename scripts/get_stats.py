# The dependencies of this script are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

import dataclasses
import logging
import os
import shutil
import tempfile
import time
from typing import Dict, List, Literal, Optional

import click
import dacite
import fsspec
import xarray as xr
import yaml

# these are auxiliary variables that exist in dataset for convenience, e.g. to do
# masking or to more easily compute vertical integrals. But they are not inputs
# or outputs to the ML model, so we don't need normalization constants for them.
common_vars = [
    "hybm",
    "hyai",
    "hyam",
    "gw",
    "area",
    "lat_bnds",
    "lon_bnds",
    "swband_bounds",
    "hybi",
    "lwband_bounds",
    "ilev",
    "lwband",
    "swband",
    "time_bnds",
]

DROP_VARIABLES = (
    [
        "land_sea_mask",
        "pressure_thickness_of_atmospheric_layer_0",
        "pressure_thickness_of_atmospheric_layer_1",
        "pressure_thickness_of_atmospheric_layer_2",
        "pressure_thickness_of_atmospheric_layer_3",
        "pressure_thickness_of_atmospheric_layer_4",
        "pressure_thickness_of_atmospheric_layer_5",
        "pressure_thickness_of_atmospheric_layer_6",
        "pressure_thickness_of_atmospheric_layer_7",
    ]
    + [f"ak_{i}" for i in range(9)]
    + [f"bk_{i}" for i in range(9)]
    + common_vars
)


DIMS = {
    "FV3GFS": ["time", "grid_xt", "grid_yt"],
    "E3SMV2": ["time", "lat", "lon"],
    "ERA5": ["time", "latitude", "longitude"],
    "CM4": ["time", "lat", "lon"],
}


def add_history_attrs(ds, input_source, start_date, end_date, n_samples):
    ds.attrs["history"] = (
        "Created by full-model/fv3gfs_data_process/get_stats.py. INPUT:"
        f" {input_source}, START_DATE: {start_date}, END_DATE: {end_date}."
    )
    ds.attrs["input_samples"] = n_samples


def copy(source: str, destination: str):
    """Copy between any two 'filesystems'. Do not use for large files.

    Args:
        source: Path to source file/object.
        destination: Path to destination.
    """
    with fsspec.open(source) as f_source:
        with fsspec.open(destination, "wb") as f_destination:
            shutil.copyfileobj(f_source, f_destination)


@dataclasses.dataclass
class StatsConfig:
    output_directory: str
    data_type: Literal["FV3GFS", "E3SMV2", "ERA5", "CM4"]
    exclude_runs: List[str] = dataclasses.field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    beaker_dataset: Optional[str] = None


@dataclasses.dataclass
class Config:
    input_type: Literal["zarr", "nc"]
    runs: Optional[Dict[str, str]]
    data_output_directory: str
    stats: StatsConfig


@click.command()
@click.argument("config_yaml", type=str)
@click.argument("run", type=int, required=False)
@click.option(
    "--debug",
    is_flag=True,
    help="If set, print some statistics instead of writing normalization coefficients.",
)
def main(config_yaml: str, run: Optional[int], debug: bool):
    """
    Compute statistics for the data processing pipeline.

    Arguments:
    config_yaml -- Path to the configuration file for the data processing pipeline.
    run -- Run index for the data processing pipeline (only used for Zarr input).
    """
    # Import dask-related things here to enable testing in environments without dask.
    import dask
    import distributed

    logging.basicConfig(level=logging.INFO)
    initial_time = time.time()
    distributed.Client(n_workers=16)

    with open(config_yaml, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    config = dacite.from_dict(data_class=Config, data=config_data)
    xr.set_options(keep_attrs=True, display_max_rows=100)

    if config.data_output_directory.endswith("/"):
        config.data_output_directory = config.data_output_directory[:-1]

    input_type = config.input_type.lower()

    if input_type == "zarr":
        if run is None:
            raise ValueError("Zarr input requires a run index.")
        run_name = list(config.runs.keys())[run]
        if run_name in config.stats.exclude_runs:
            logging.info(f"Skipping run {run_name}")
            return
        input_source = os.path.join(config.data_output_directory, run_name)
        logging.info(f"Reading Zarr data from {input_source}.zarr" )
        with dask.config.set({"array.chunk-size": "128MiB"}):
            ds = xr.open_zarr(input_source+ ".zarr", chunks={"time": "auto"})

    elif input_type == "nc":
        input_source =  config.data_output_directory + "/" + "*.nc"
        logging.info(f"Reading NetCDF files from {input_source}")
        with dask.config.set({"array.chunk-size": "128MiB"}):
            ds = xr.open_mfdataset(input_source,
                                    chunks={"time": 128},   # one file = one chunk
                                    parallel=True
                                )

    else:
        raise ValueError(f"Unsupported input_type: {input_type}")
    logging.info("Data Loading Done")
    ds = ds.drop_vars(DROP_VARIABLES, errors="ignore")    
    #ds = ds.sel(time=slice(config.stats.start_date, config.stats.end_date))

    print(ds)

    ds = ds.resample(time="1MS").mean()
    
    dims = DIMS[config.stats.data_type]

    # Explicitly compute the statistics here, since xarray does not support
    # writing netCDFs with the scipy engine with the distributed scheduler.
    # There is no harm to computing here versus later, since the end result is
    # not something memory intensive.
    centering = ds.mean(dim=dims).compute()
    logging.info("Computed centering")
    scaling_full_field = ds.std(dim=dims).compute()
    logging.info("Computed scaling_full_field")
    scaling_residual = ds.diff("time").std(dim=dims).compute()
    logging.info("Computed scaling_resiudal")
    time_means = ds.mean(dim="time").compute()
    logging.info("Computed time_means")

    for dataset in [
        centering,
        scaling_full_field,
        scaling_residual,
        time_means,
    ]:
        n_samples = len(ds.time)
        add_history_attrs(
            dataset,
            input_source,
            config.stats.start_date,
            config.stats.end_date,
            n_samples,
        )

    if debug:
        normed_data = (ds - centering) / scaling_full_field
        logging.info(f"Average of normed data: {normed_data.mean(dim=dims).compute()}")
        logging.info(
            f"Standard deviation of normed data: {normed_data.std(dim=dims).compute()}"
        )
        all_var_stddev = normed_data.to_array().std(dim=["variable"] + dims)
        logging.info(
            f"Standard deviation computed over all variables: {all_var_stddev.values}"
        )
    else:
        if input_type == "zarr":
            out_dir = os.path.join(config.stats.output_directory, run_name)
        else:
            out_dir = config.stats.output_directory

        if out_dir.startswith("gs:"):
            temp_dir = tempfile.TemporaryDirectory()
            local_dir = temp_dir.name
            remote_dir: Optional[str] = out_dir
        else:
            os.makedirs(out_dir, exist_ok=True)
            local_dir = out_dir
            remote_dir = None

        centering.to_netcdf(os.path.join(local_dir, "centering.nc"))
        if remote_dir is not None:
            copy(
                os.path.join(local_dir, "centering.nc"),
                remote_dir + "/centering.nc",
            )
        scaling_full_field.to_netcdf(os.path.join(local_dir, "scaling-full-field.nc"))
        if remote_dir is not None:
            copy(
                os.path.join(local_dir, "scaling-full-field.nc"),
                remote_dir + "/scaling-full-field.nc",
            )
        scaling_residual.to_netcdf(os.path.join(local_dir, "scaling-residual.nc"))
        if remote_dir is not None:
            copy(
                os.path.join(local_dir, "scaling-residual.nc"),
                remote_dir + "/scaling-residual.nc",
            )
        time_means.to_netcdf(os.path.join(local_dir, "time-mean.nc"))
        if remote_dir is not None:
            copy(
                os.path.join(local_dir, "time-mean.nc"),
                remote_dir + "/time-mean.nc",
            )

    total_time = time.time() - initial_time
    logging.info(f"Total time for computing stats: {total_time:0.2f} seconds.")


if __name__ == "__main__":
    main()
