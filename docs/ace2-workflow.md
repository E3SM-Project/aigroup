# ACE2-ERA5 Training Workflow

This guide provides a complete workflow for training the ACE (AI2 Climate Emulator) model using the ACE2-ERA5 dataset.

## Overview

ACE is a machine learning model for climate emulation developed by AI2. This workflow will guide you through setting up the environment and running training on a GPU compute node.

## Prerequisites

- Access to a compute cluster with GPU nodes (e.g., pm-gpu)
- Storage space for the dataset (ACE2-ERA5)
- Network access to clone repositories

## Resources

- **Code Repository**: [E3SM-Project/ace](https://github.com/E3SM-Project/ace)
- **Dataset**: [ACE2-ERA5 on Hugging Face](https://huggingface.co/allenai/ACE2-ERA5)
- **Documentation**: [ACE Training Configuration Guide](https://ai2-climate-emulator.readthedocs.io/en/latest/training_config.html)

## Setup Instructions

### 1. Clone the Code Repository

Start by cloning the ACE repository and checking out the main branch:

```console
git clone https://github.com/E3SM-Project/ace.git
cd ace
```

### 2. Download the Dataset

Clone the ACE2-ERA5 dataset from Hugging Face:

```console
git clone https://huggingface.co/allenai/ACE2-ERA5
```

!!! note "Dataset Size"
    The ACE2-ERA5 dataset is large. Ensure you have sufficient storage space before cloning.
    Also, if you run into issues related to git lfs, you may need to install that.

### 3. Install uv Package Manager

Install the `uv` package manager, which is used to manage Python dependencies:

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, you may need to restart your console or source your profile to use `uv`.

### 4. Configure uv Cache

Set up the uv cache directory in an accessible location with sufficient storage:

```console
mkdir -p "$PSCRATCH/.cache/uv"
export UV_CACHE_DIR="$PSCRATCH/.cache/uv"
```

!!! tip "Cache Location"
    Adjust `$PSCRATCH` to match your system's scratch directory path. This ensures the cache is stored in a location with adequate space.

### 5. Pin Python Version

Pin the Python version to 3.11:

```console
uv python pin 3.11
```

### 6. Pin Setuptools Version

To avoid deprecation warnings related to `pkg_resources`, pin setuptools to a version below 81:

```console
uv add --dev 'setuptools<81'
```

!!! warning "pkg_resources Deprecation"
    The ACE codebase and some of its dependencies use `pkg_resources`, which is deprecated in setuptools 81+. Setuptools plans to remove `pkg_resources` as early as November 30, 2025. Until the dependencies are updated to use `importlib.resources` or `importlib.metadata`, you must pin setuptools to version <81 to avoid warnings and future breakage.

## Running Training

### 1. Request a GPU Compute Node

Request an interactive GPU node on your cluster:

```console
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --account=e3sm_g
```

!!! note "Account Settings"
    Adjust the `--account` parameter to match your allocation account.

### 2. Prepare Training Configuration

Create a training configuration file named `config-train.yaml` in the repository root. You can start with a template from the [ACE training configuration documentation](https://ai2-climate-emulator.readthedocs.io/en/latest/training_config.html), or use the sample below:

??? example "Sample Configuration (`config-train.yaml`)"
    ```yaml
    experiment_dir: /path/to/your/ACE2-ERA5/train_output
    save_checkpoint: true
    validate_using_ema: true
    max_epochs: 80
    n_forward_steps: 1
    inference:
      n_forward_steps: 300  # ~75 days (adjust based on your needs)
      forward_steps_in_memory: 1
      loader:
        start_indices:
          first: 0
          n_initial_conditions: 4
          interval: 300  # adjusted to fit within dataset
        dataset:
          data_path: /path/to/your/ACE2-ERA5/training_validation_data/training_validation
        num_data_workers: 4
    logging:
      log_to_screen: true
      log_to_wandb: false
      log_to_file: true
      project: ace
      entity: your_wandb_entity
    train_loader:
      batch_size: 4
      num_data_workers: 2
      prefetch_factor: 2
      dataset:
        concat:
          - data_path: /path/to/your/ACE2-ERA5/training_validation_data/training_validation
    validation_loader:
      batch_size: 4
      num_data_workers: 2
      prefetch_factor: 2
      dataset:
        data_path: /path/to/your/ACE2-ERA5/training_validation_data/training_validation
        subset:
          step: 5
    optimization:
      enable_automatic_mixed_precision: false
      lr: 0.0001
      optimizer_type: AdamW
      # can also set kwargs: fused: true for performance if using GPU
    stepper:
      loss:
        type: MSE
      step:
        type: single_module
        config:
          builder:
            type: SphericalFourierNeuralOperatorNet
            config:
              embed_dim: 16
              filter_type: linear
              hard_thresholding_fraction: 1.0
              use_mlp: true
              normalization_layer: instance_norm
              num_layers: 2
              operator_type: dhconv
              scale_factor: 1
              separable: false
          normalization:
            network:
              global_means_path: /path/to/your/ACE2-ERA5/training_validation_data/normalization/centering.nc
              global_stds_path: /path/to/your/ACE2-ERA5/training_validation_data/normalization/scaling-full-field.nc
            loss:
              global_means_path: /path/to/your/ACE2-ERA5/training_validation_data/normalization/centering.nc
              global_stds_path: /path/to/your/ACE2-ERA5/training_validation_data/normalization/scaling-residual.nc
          in_names:
          - land_fraction
          - ocean_fraction
          - sea_ice_fraction
          - DSWRFtoa
          - HGTsfc
          - PRESsfc
          - surface_temperature
          - air_temperature_0 # _0 denotes the top most layer of the atmosphere
          - air_temperature_1
          - air_temperature_2
          - air_temperature_3
          - air_temperature_4
          - air_temperature_5
          - air_temperature_6
          - air_temperature_7
          - specific_total_water_0
          - specific_total_water_1
          - specific_total_water_2
          - specific_total_water_3
          - specific_total_water_4
          - specific_total_water_5
          - specific_total_water_6
          - specific_total_water_7
          - eastward_wind_0
          - eastward_wind_1
          - eastward_wind_2
          - eastward_wind_3
          - eastward_wind_4
          - eastward_wind_5
          - eastward_wind_6
          - eastward_wind_7
          - northward_wind_0
          - northward_wind_1
          - northward_wind_2
          - northward_wind_3
          - northward_wind_4
          - northward_wind_5
          - northward_wind_6
          - northward_wind_7
          out_names:
          - PRESsfc
          - surface_temperature
          - air_temperature_0
          - air_temperature_1
          - air_temperature_2
          - air_temperature_3
          - air_temperature_4
          - air_temperature_5
          - air_temperature_6
          - air_temperature_7
          - specific_total_water_0
          - specific_total_water_1
          - specific_total_water_2
          - specific_total_water_3
          - specific_total_water_4
          - specific_total_water_5
          - specific_total_water_6
          - specific_total_water_7
          - eastward_wind_0
          - eastward_wind_1
          - eastward_wind_2
          - eastward_wind_3
          - eastward_wind_4
          - eastward_wind_5
          - eastward_wind_6
          - eastward_wind_7
          - northward_wind_0
          - northward_wind_1
          - northward_wind_2
          - northward_wind_3
          - northward_wind_4
          - northward_wind_5
          - northward_wind_6
          - northward_wind_7
          - LHTFLsfc
          - SHTFLsfc
          - PRATEsfc
          - ULWRFsfc
          - ULWRFtoa
          - DLWRFsfc
          - DSWRFsfc
          - USWRFsfc
          - USWRFtoa
          - tendency_of_total_water_path_due_to_advection
    ```

**Important**: Make sure to update the following in your `config-train.yaml`:

- `experiment_dir`: Set this to a writable directory where training outputs will be saved
- `data_path`: Point this to your downloaded ACE2-ERA5 dataset location

!!! tip "Fast Iteration"
    For faster iteration during initial testing, consider:

    - Reducing the number of training epochs
    - Using a smaller batch size
    - Limiting the dataset size

### 3. Launch Training

From the repository root, launch the training job using `torchrun`:

```console
uv run torchrun --nproc_per_node=4 -m fme.ace.train config-train.yaml
```

This command will:

- Use `uv run` to manage dependencies automatically
- Launch `torchrun` with 4 processes (one per GPU)
- Execute the training module with your configuration

## Training Parameters

The `torchrun` command accepts several parameters:

- `--nproc_per_node=4`: Number of processes per node (typically matches the number of GPUs)
- `-m fme.ace.train`: The Python module to run
- `config-train.yaml`: Your training configuration file

## Monitoring Training

During training, monitor:

- GPU utilization: Use `nvidia-smi` to check GPU usage
- Training logs: Check the output directory specified in `experiment_dir`
- Checkpoints: Models will be saved periodically based on your configuration

## Troubleshooting

### Common Issues

**pkg_resources Deprecation Warnings**

If you see warnings like:
```
UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
```

This occurs when setuptools version 81 or higher is installed. The ACE codebase and its dependencies still use `pkg_resources`, which is deprecated. To resolve this:

- Pin setuptools to version <81: `uv add --dev 'setuptools<81'`
- Alternatively, suppress the warnings (not recommended) until dependencies are updated
- Note: pkg_resources is scheduled for removal in setuptools, potentially as early as November 30, 2025

**Out of Memory Errors**

- Reduce batch size in `config-train.yaml`
- Decrease model size parameters
- Use fewer GPUs with `--nproc_per_node`

**Cache Directory Issues**

- Ensure `UV_CACHE_DIR` has sufficient space
- Check write permissions on the cache directory

**Module Import Errors**

- Verify Python version is pinned to 3.11
- Ensure you're running from the repository root
- Check that dependencies are properly installed by `uv`

## Additional Resources

- [ACE Documentation](https://ai2-climate-emulator.readthedocs.io/)
- [uv Documentation](https://github.com/astral-sh/uv)
- [PyTorch Distributed Training](https://pytorch.org/docs/stable/distributed.html)
