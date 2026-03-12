---
title: "E3SM Data Processing Pipeline"
description: "Process E3SM output (h0/h1/h2) into ACE-ready ML datasets: full, forcing, and initial conditions."
---

# E3SM Data Processing Pipeline

This pipeline processes **E3SM climate model output files** to create datasets suitable for **AI/ML training** with the **ACE (Ai2 Climate Emulator)** model.

---

## 1. Remapping - `remapping_parallel_EAM.sh`

Submits a parallel SLURM job to regrid E3SM h0/h1/h2 NetCDF files using `ncremap` and a specified mapping file.

**Features:**
- Processes files in multi-year chunks
- Validates inputs
- Runs MPI-based remapping
- Writes outputs to destination directory
- Verifies input/output file counts match

### How to Run on Perlmutter

```bash
sbatch remapping_parallel_EAM.sh
```

---

## 2. E3SM Data Processing for Inference - `process_e3sm_data_ai.py`

Processes E3SM climate model output files to create ACE-compatible datasets.


### Workflow

```text
Input E3SM Files (h1, h2, h0)
    ↓
[1] Group files by month
    ↓
[2] Open & filter to target month
    ↓
[3] Select required variables
    ↓
[4] Merge h1 + h2 + h0
    ↓
[5] Compute derived variables
    ├── Pressure thickness
    ├── Total water
    ├── Precipitation rate
    ├── Radiative fluxes
    └── Water path tendencies
    ↓
[6] Vertical coarsening
    ├── Mass-weighted averaging
    └── Compute ak/bk coefficients
    ↓
[7] Convert to float32
    ↓
[8] Create output datasets
    ├── Full dataset
    ├── Forcing dataset
    └── Initial condition dataset
    ↓
[9] Save NetCDF files
    ├── YYYY-MM.nc
    ├── YYYY-MM.forcing.nc
    └── YYYY-MM.ic.nc
```
### How to Run
```bash
python process_e3sm_data_ai.py --config config_process_e3sm_data_ai.yaml
```

## 3. E3SM Data Processing for Training - ` `
