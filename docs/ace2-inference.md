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

```yaml
experiment_dir: /pscratch/sd/m/mahf708/ACE2-EAMv3/test1
n_forward_steps: 1000
forward_steps_in_memory: 100
checkpoint_path:  /pscratch/sd/m/mahf708/ACE2-EAMv3/ace2_EAMv3_ckpt.tar
initial_condition:
  path: /pscratch/sd/m/mahf708/ACE2-EAMv3/initial_conditions/1971010100.nc
  start_indices:
    n_initial_conditions: 2
    first: 0
    interval: 1
forcing_loader:
  dataset:
    data_path: /pscratch/sd/m/mahf708/ACE2-EAMv3/forcing_data/
  num_data_workers: 2
logging:
  log_to_screen: true
  log_to_wandb: false
  log_to_file: true

data_writer:
  save_prediction_files: true
```

!!! todo "Annotate Config Details"
    Add descriptions for each configuration parameter, explaining what they control and recommended values.

Run the inference using the following command:

```console
uv run python -m fme.ace.inference config-inference.yaml
```

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
> ncdump -h /pscratch/sd/m/mahf708/ACE2-EAMv3/test1/autoregressive_predictions.nc 
netcdf autoregressive_predictions {
dimensions:
        time = UNLIMITED ; // (1000 currently)
        sample = 2 ;
        lat = 180 ;
        lon = 360 ;
variables:
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

!!! todo "Annotate Results Details"
    Provide more details about the output files, including variable definitions and units.
