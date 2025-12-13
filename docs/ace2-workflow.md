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

```shell
git clone https://github.com/E3SM-Project/ace.git
cd ace
```

### 2. Download the Dataset

Clone the ACE2-ERA5 dataset from Hugging Face:

```shell
git clone https://huggingface.co/allenai/ACE2-ERA5
```

!!! note "Dataset Size"
    The ACE2-ERA5 dataset is large. Ensure you have sufficient storage space before cloning.
    Also, if you run into issues related to git lfs, you may need to install that.

### 3. Install uv Package Manager

Install the `uv` package manager, which is used to manage Python dependencies:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, you may need to restart your shell or source your profile to use `uv`.

### 4. Configure uv Cache

Set up the uv cache directory in an accessible location with sufficient storage:

```shell
mkdir -p "$PSCRATCH/.cache/uv"
export UV_CACHE_DIR="$PSCRATCH/.cache/uv"
```

!!! tip "Cache Location"
    Adjust `$PSCRATCH` to match your system's scratch directory path. This ensures the cache is stored in a location with adequate space.

### 5. Pin Python Version

Pin the Python version to 3.11:

```shell
uv python pin 3.11
```

## Running Training

### 1. Request a GPU Compute Node

Request an interactive GPU node on your cluster:

```shell
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --account=e3sm_g
```

!!! note "Account Settings"
    Adjust the `--account` parameter to match your allocation account.

### 2. Prepare Training Configuration

Create a training configuration file named `config-train.yaml` in the repository root. You can start with a template from the [ACE training configuration documentation](https://ai2-climate-emulator.readthedocs.io/en/latest/training_config.html).

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

```shell
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

## Next Steps

After successful training:
- Evaluate model performance using validation scripts
- Export trained models for inference
- Visualize training metrics and results
- Fine-tune hyperparameters based on initial results

## Additional Resources

- [ACE Documentation](https://ai2-climate-emulator.readthedocs.io/)
- [uv Documentation](https://github.com/astral-sh/uv)
- [PyTorch Distributed Training](https://pytorch.org/docs/stable/distributed.html)
