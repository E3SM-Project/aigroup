# ACE2 Inference Tutorial

This guide walks you through running inference using the pre-trained [ACE2-EAMv3](https://huggingface.co/allenai/ACE2-EAMv3) model.

## Prerequisites for this guide

- [uv](https://github.com/astral-sh/uv) installed to set up the environment, including Py
Torch. See our [Python Environment Setup](python-envs.md) for more details.
- Access to the internet to clone repositories

## Steps

### 1. Clone the ACE Repository

Clone the ACE repository from GitHub:

```console
git clone https://github.com/E3SM-Project/ace
```

### 2. Clone the Model Repository

Clone the ACE2-EAMv3 model repository from Hugging Face:

```console
git clone https://huggingface.co/allenai/ACE2-EAMv3
```

!!! note "git lfs"
    If you run into issues related to git lfs, you may need to install that.

### 3. Run Inference

Navigate to the `ace` repository directory:

```console
cd ace
```

Create a configuration file named `config-inference.yaml` with the following content. Make sure to update the paths to match your environment.

``` { .yaml .annotate }
experiment_dir: /pscratch/sd/m/mahf708/ACE2-EAMv3/test1 # (1)!
n_forward_steps: 1458 # (2)!
forward_steps_in_memory: 80 # (3)!
checkpoint_path: /pscratch/sd/m/mahf708/ACE2-EAMv3/ace2_EAMv3_ckpt.tar # (4)!
initial_condition: # (5)!
  path: /pscratch/sd/m/mahf708/ACE2-EAMv3/initial_conditions/1971010100.nc
  start_indices:
    n_initial_conditions: 2
    first: 0
    interval: 1
forcing_loader: # (6)!
  dataset:
    data_path: /pscratch/sd/m/mahf708/ACE2-EAMv3/forcing_data/
  num_data_workers: 2
logging: # (7)!
  log_to_screen: true
  log_to_wandb: false
  log_to_file: true
data_writer: # (8)!
  save_prediction_files: true
```

1. **Output directory** — All inference outputs (predictions, diagnostics, logs) are saved here. Create this directory before running.
2. **Number of forward steps** — Total timesteps to run. Each step is 6 hours, so 1458 steps ≈ 365 days. See note below on limitation.
3. **Steps in memory** — Batch size for GPU memory. Lower this if you run into OOM errors. 80 is a good default.
4. **Model checkpoint** — Path to the pretrained ACE2-EAMv3 weights (`.tar` file from Hugging Face).
5. **Initial conditions** — Starting atmospheric state. `n_initial_conditions` runs multiple ensemble members; `first` and `interval` control which samples to use from the IC file.
6. **Forcing data** — External forcing (SST, solar, GHGs, etc.). The loader reads Zarr/NetCDF files from this path. `num_data_workers` controls parallel I/O.
7. **Logging options** — `log_to_screen` prints progress; `log_to_wandb` sends metrics to Weights & Biases (requires login); `log_to_file` saves to `inference_out.log`.
8. **Output writer** — Set `save_prediction_files: true` to write NetCDF outputs. Set to `false` for validation-only runs. Additionally, one could `names: [T_4, T_5]` to request only `T_4` and `T_5` in the output.

!!! tip "maximum steps"
    Note that this is limited by the temporal length of the forcing data
    (in the example above, a year; see forcing_data) and the specifics of the initial conditions
    (in the example above, 2 seperated by a single time step starting from 0).
    That's why we have an offset of 2 steps from a full year in the prediction.
    If we have one initial conditions, then the number of forward steps would be 1459.
    The general formula is: `max steps allowed = length of data - (first + interval * (n_initial_conditions-1))`

Run the inference using the following command:

```console
uv run python -m fme.ace.inference config-inference.yaml
```

!!! tip "compute node"
    The above command takes about 10 minutes on a single compute node on pm-gpu (4xA100).
    The command to get a compute pm-gpu compute node is: 
    ```console
    salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --account=e3sm_g
    ```

!!! tip "uv cache"
    Sometimes, you will need to the enviornment variable `UV_CACHE_DIR`, e.g., on NERSC, `export UV_CACHE_DIR="$PSCRATCH/.cache/uv"`

### 4. Results

The results will be saved in the `experiment_dir` specified in the config file. The output directory structure will look like this:

```console
> ls /pscratch/sd/m/mahf708/ACE2-EAMv3/test1 -1 
annual_diagnostics.nc
autoregressive_predictions.nc
autoregressive_target.nc
config.yaml
inference_out.log
initial_condition.nc
mean_diagnostics.nc
monthly_mean_predictions.nc
monthly_mean_target.nc
restart.nc
time_mean_diagnostics.nc  
```

where the autoregressive_predictions.nc file has the following header:

```console
ncdump -h /pscratch/sd/m/mahf708/ACE2-EAMv3/test1/autoregressive_predictions.nc
netcdf autoregressive_predictions {
dimensions:
        time = UNLIMITED ; // (1458 currently)
        sample = 2 ;
        lat = 180 ;
        lon = 360 ;
variables:
        int64 time(time) ;
                time:units = "microseconds" ;
        int64 init_time(sample) ;
                init_time:units = "microseconds since 1970-01-01 00:00:00" ;
                init_time:calendar = "noleap" ;
        ...
```

These are the available variables:

```console
> ncdump -h /pscratch/sd/m/mahf708/ACE2-EAMv3/test1/autoregressive_predictions.nc | grep "float\|int64" | awk '{print $2}' | cut -d'(' -f1
time
init_time
valid_time
lat
lon
TS
net_energy_flux_sfc_into_atmosphere
T_4
V_2
surface_pressure_due_to_dry_air_absolute_tendency
T_3
T_0
T_6
V_4
U_3
U_4
tendency_of_total_water_path_due_to_advection
specific_total_water_3
specific_total_water_4
U_1
V_7
U_5
FLUT
surface_precipitation_rate
surface_upward_longwave_flux
top_of_atmos_upward_shortwave_flux
specific_total_water_0
T_7
U_2
net_energy_flux_into_atmospheric_column
specific_total_water_1
specific_total_water_6
V_0
total_water_path
V_1
V_6
FLDS
FSDS
PS
total_energy_ace2_path
T_1
V_3
surface_pressure_due_to_dry_air
U_6
specific_total_water_5
V_5
T_5
U_0
SHFLX
LHFLX
specific_total_water_7
surface_upward_shortwave_flux
T_2
U_7
specific_total_water_2
total_energy_ace2_path_tendency
total_water_path_budget_residual
implied_tendency_of_total_energy_ace2_path_due_to_advection
net_energy_flux_toa_into_atmosphere
```

## Remaining tasks

- [ ] Prepare forcing data for longer time period (e.g., 10 years)
- [ ] Explain the variables and files produces
  - [ ] a restart.nc file is recorded. How do we perform a restart run?
  - [ ] more information about the variables and levels
- [ ] Explore performance space, and producing larger ensembles
