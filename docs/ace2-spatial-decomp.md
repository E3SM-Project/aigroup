# ACE2-ERA5 Training Workflow (Spatial Decomposition)

This guide provides a complete workflow for training the ACE (AI2 Climate Emulator) model using the ACE2-ERA5 dataset with **spatial decomposition** (model-parallel) across multiple GPUs and nodes.

## Overview

This is the spatial-decomposition counterpart to the [vanilla ACE2-ERA5 training workflow](ace2-workflow.md). The setup is identical except that:

- Training runs across multiple nodes (so we use `srun torchrun` with a rendezvous endpoint).
- The model is split spatially across GPUs along the height and width dimensions via the `model` distributed backend in FME.
- A few `FME_DISTRIBUTED_*` environment variables control the decomposition.

If you have not already worked through the vanilla guide, skim it first — the prerequisites, dataset download, `uv` setup, and Python/setuptools pinning are the same and are not repeated here.

## Prerequisites

- Access to a multi-node GPU cluster (e.g., pm-gpu)
- Storage space for the ACE2-ERA5 dataset
- Completed steps 1–6 of the [vanilla workflow](ace2-workflow.md#setup-instructions) (clone the code, download the data, install `uv`, configure cache, pin Python 3.11, pin `setuptools<81`)

## Resources

- **Code Repository**: [E3SM-Project/ace](https://github.com/E3SM-Project/ace)
- **Dataset**: [ACE2-ERA5 on Hugging Face](https://huggingface.co/allenai/ACE2-ERA5)
- **Documentation**: [ACE Training Configuration Guide](https://ai2-climate-emulator.readthedocs.io/en/latest/training_config.html)

## How Spatial Decomposition Works

The FME training code supports three distributed backends, selected via the `FME_DISTRIBUTED_BACKEND` environment variable:

- `torch` (default): pure data parallelism across all GPUs
- `model`: spatial (model) parallelism, splitting the grid by height (`H`) and width (`W`)
- `none`: forces non-distributed execution

When `FME_DISTRIBUTED_BACKEND=model`, you must also set `FME_DISTRIBUTED_H` and `FME_DISTRIBUTED_W`. The remaining processes (after spatial decomposition) form a data-parallel dimension.

### Sizing constraints

Let `NPROCS` be the total number of GPUs in the job (i.e. `nnodes * nproc_per_node`), and let

```
DATA_DIM = NPROCS / (FME_DISTRIBUTED_H * FME_DISTRIBUTED_W)
```

Two constraints must be satisfied:

1. `NPROCS` must be divisible by `FME_DISTRIBUTED_H * FME_DISTRIBUTED_W` (i.e. `DATA_DIM` is an integer).
2. `train_loader.batch_size` must be divisible by `DATA_DIM`.

!!! example "Sizing example"
    For 2 nodes × 4 GPUs/node = 8 GPUs with `H=2, W=2`:

    - `DATA_DIM = 8 / (2 * 2) = 2`
    - `batch_size = 4` works (4 % 2 == 0).

## Running Training

### 1. Request multiple GPU nodes

Request an interactive allocation with the number of nodes you need. For 2 nodes:

```console
salloc --nodes 2 --qos interactive --time 04:00:00 --constraint gpu --account=e3sm_g --gpus-per-node=4
```

!!! note "Account settings"
    Adjust `--account` to match your allocation. Bump `--nodes` up for larger spatial layouts.

### 2. Prepare Training Configuration

Create a training configuration file named `config-train.yaml` in the repository root. The config itself is essentially the same as the vanilla one — spatial decomposition is configured entirely through environment variables, not the YAML. A working example follows.

??? example "Sample Configuration (`config-train.yaml`)"
    ```yaml
    experiment_dir: /path/to/your/ACE2-ERA5/train_output_sp
    save_checkpoint: true
    validate_using_ema: false
    max_epochs: 2
    # inference:
    #   n_forward_steps: 300
    #   forward_steps_in_memory: 1
    #   loader:
    #     start_indices:
    #       first: 0
    #       n_initial_conditions: 4
    #       interval: 300
    #     dataset:
    #       data_path: /path/to/your/ACE2-ERA5/training_validation_data/training_validation
    #     num_data_workers: 4
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
    stepper_training:
      loss:
        type: MSE
      n_forward_steps: 1
    stepper:
      step:
        type: single_module
        config:
          builder:
            type: NoiseConditionedSFNO
            config:
              embed_dim: 16
              filter_type: linear
              use_mlp: true
              num_layers: 2
              operator_type: dhconv
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

- `experiment_dir`: a writable directory where training outputs will be saved
- `data_path`: your downloaded ACE2-ERA5 dataset location
- `train_loader.batch_size`: must be divisible by `DATA_DIM = NPROCS / (H * W)` (see [sizing constraints](#sizing-constraints))

### 3. Set the spatial-decomposition environment variables

Before launching, export the FME backend variables that control the decomposition:

```console
export FME_DISTRIBUTED_BACKEND=model   # default is "torch" (data-parallel only)
export FME_DISTRIBUTED_H=2             # split height into 2
export FME_DISTRIBUTED_W=2             # split width into 2
```

For the 2-node × 4-GPU example above, this gives `DATA_DIM = 8 / (2*2) = 2`, which is compatible with `batch_size: 4`.

### 4. Launch Training

Pick the head node from the SLURM allocation and launch with `srun torchrun` using a `c10d` rendezvous on that node:

```console
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun uv run torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${HEAD_NODE}:29500" \
    -m fme.ace.train config-train.yaml
```

This command will:

- Use `uv run` to manage dependencies automatically
- Use `srun` so `torchrun` is launched on every node in the allocation
- Use a `c10d` rendezvous pinned to the first (head) node
- Run with 4 processes per node (one per GPU), for a total of `nnodes * 4` ranks
- Pick up `FME_DISTRIBUTED_BACKEND=model` (plus `H`/`W`) from the environment and split the model spatially

!!! tip "Going back to plain data-parallel"
    To run the same job as pure data parallelism (no spatial split), simply unset (or set to `torch`) `FME_DISTRIBUTED_BACKEND` and drop `FME_DISTRIBUTED_H`/`FME_DISTRIBUTED_W`. Everything else, including the `srun torchrun ...` launcher, stays the same.
