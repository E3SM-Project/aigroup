# Python Environment Setup

Quick guide for setting up Python environments to run AI group workflows.
There are several options for setting up Python environments, including

- [uv](https://docs.astral.sh/uv/)
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (and other conda variants)
- [pip](https://pip.pypa.io/en/stable/)

In this guide, we will only highlight uv,
but users can achieve the same results with other tools.

!!! warning "only a quick start"
    This is simply a starting guide, and isn't meant to be exhaustive or performant.


---

## Recommended option: uv

The [uv](https://docs.astral.sh/uv/) tool is a fast, modern Python package manager. It's significantly faster than pip and handles virtual environments elegantly.

### Install uv

```console
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone ACE and install env manually

```console
# Clone ACE
git clone https://github.com/E3SM-Project/ace.git
cd ace

# Create venv with specific Python version
uv venv --python 3.11

# Activate
source .venv/bin/activate

# Install dependencies from requirements.txt
uv pip install -r requirements.txt

# Install ACE in editable mode
uv pip install -e .
```

### Clone ACE and run automatically

Instead of manual installation, you can clone ACE and then let uv handle the rest:

```console
# Clone ACE
git clone https://github.com/E3SM-Project/ace.git
cd ace

uv run python -m fme.ace.inference scratch/config-inference.yaml
```

The last command starting with `uv run` ([documentation](https://docs.astral.sh/uv/reference/cli/#uv-run)) ensures that the specified environment in the ACE repo is set and the trailing instruction (`python -m fme.ace.inference scratch/config-inference.yaml`) happens inside of it. This environment is still saved in `.venv` in the root of the repo.

!!! tip "uv cache" 
    Sometimes, you will need to the enviornment variable `UV_CACHE_DIR`, e.g., on NERSC, `export UV_CACHE_DIR="$PSCRATCH/.cache/uv"`


## Last words:

In general, there's no one-size-fits-all solution for setting up Python environments.
It really depends on your specific needs and preferences.
Because Python is ubiquitous, the user must decide which tool to use for their specific needs, and learn to use it effectively for their needs.
