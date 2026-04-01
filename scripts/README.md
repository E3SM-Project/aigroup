# E3SM Data Processing Pipeline

This folder contains scripts for processing E3SM atmospheric model output, including regridding, preparing AI-ready datasets, converting to Zarr format, and computing statistics for machine learning.

## Pipeline Overview

1. **Regrid E3SM output to regular lat-lon grid**  
   (`remapping_parallel_EAM_v2.sh`)
2. **Process regridded data into AI-ready monthly NetCDFs**  
   (`process_e3sm_data_ai_month.py`)
3. **Convert monthly NetCDFs to partitioned Zarr store**  
   (`to_zarr.py`)
4. **Compute statistics (mean, std, etc.) for ML normalization**  
   (`get_stats.py`)

---

## 1. Regridding: `remapping_parallel_EAM_v2.sh`

This SLURM batch script uses `ncremap` to regrid E3SM output files to a regular lat-lon grid in parallel.

- **Edit the script** to set:
  - `SRC_PATH`: Directory with original E3SM output files
  - `DES_PATH`: Output directory for regridded files
  - `MAP`: Path to the mapping file (e.g., `map_ne30pg2_to_gaussian_180by360.nc`)
  - `START_YEAR`, `END_YEAR`: Year range to process

- **Submit the job**:
  ```bash
  sbatch remapping_parallel_EAM_v2.sh
  ```

- **Output**: Regridded NetCDF files in `$DES_PATH`

---

## 2. Process AI Data: `process_e3sm_data_ai_month.py`

This script merges, computes derived variables, coarsens vertical levels, and outputs monthly AI-ready NetCDFs for forcing and training.

  `config_process_e3sm_data_ai_month_3.yaml`  
  - Set `data_paths` to point to your regridded files
  - Set `output_directory` for processed files
  - Adjust year/month range and variables as needed

  ```bash
  python process_e3sm_data_ai_month.py --config config_process_e3sm_data_ai_month.yaml
  ```
  - Optional: `--start-year`, `--end-year`, `--start-month`, `--end-month` to override config
  - Add `--debug` for verbose logging

  - `forcingdata/` and `traindata/` subfolders under your `output_directory`, each with monthly NetCDFs

**Workflow:**

Key Functions Used:

1. `group_files_by_month` – Group input files by month
2. `open_files_for_month` – Open and filter files for a specific month
3. `select_variables` – Select required variables from datasets
4. `compute_pressure_thickness_mid` – Compute pressure thickness
5. `compute_specific_total_water_e3sm` – Compute total water
6. `compute_surface_precipitation_rate` – Compute precipitation rate
7. `compute_rad_fluxes` – Compute radiative fluxes
8. `compute_total_water_path` – Compute total water path
9. `compute_twp_tendency` – Compute water path tendencies
10. `compute_twp_advective_tendency` – Compute advective tendency
11. `roundtrip_filter` – Apply spherical harmonics filtering to selected variables (if enabled in config via `roundtrip_fraction_kept`)
12. `compute_vertical_coarsening` – Mass-weighted vertical coarsening
13. `compute_coarse_ak_bk` – Compute ak/bk coefficients
14. `convert_to_float32` – Convert all variables to float32
15. `create_forcing_dataset` – Create forcing dataset
16. `create_traindata_dataset` – Create traindata dataset
---


## 3. Convert to Zarr - `to_zarr.py`

This script combines monthly NetCDFs into a partitioned Zarr store for efficient ML training.

**How to Run:**
```bash
python -u to_zarr.py \
  --monthly-files /path/to/output/traindata \
  --store-name /path/to/output/traindata_YYYY_YYYY.zarr \
  --partitions years (years of dataset is usually ideal)
```
- `/path/to/output/traindata`: Directory containing monthly NetCDFs
- `/path/to/output/traindata_YYYY_YYYY.zarr`: Output Zarr store
- `--partitions`: Number of partitions along the time dimension

**Output:**
- Partitioned Zarr store (e.g., `traindata_YYYY_YYYY.zarr`)

---

## 4. Compute Statistics: `get_stats.py`


This script computes mean, std, and other statistics for normalization, using the Zarr or NetCDF data.

**Note:** The Zarr file name used for statistics is the one specified in the `runs` section of your stats config file. For example, if your config contains:

```yaml
runs:
  file_name: ""
```
then the Zarr store should be named `file_name.zarr` in the appropriate output directory.

- **Edit the config**:  
  `e3sm_stats_config.yaml`  
  - Set `data_output_directory` to your processed data
  - Set `stats.output_directory` for stats output
  - Set `input_type` to `zarr` or `nc` as appropriate

- **Run the script**:
  ```bash
  python get_stats.py e3sm_stats_config.yaml <run_index>
  ```
  - For Zarr: `<run_index>` is the index of the run in the config (usually 0)
  - For NetCDF: omit `<run_index>`

- **Output**:  
  - NetCDF files with centering, scaling, and time-mean statistics in the specified stats output directory

---

## Notes

- All scripts require a working Python environment with the necessary dependencies (see code comments for details).
- Adjust paths and config files as needed for your data and system.
- For large datasets, run scripts on a compute cluster with sufficient resources.
