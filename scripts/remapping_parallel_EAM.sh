#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH -C cpu
#SBATCH -A e3sm
#SBATCH -q debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -J ncremap_e3sm
#SBATCH -o logs/job_log_%j.out

# ------------------ Environment ------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.sh


set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ------------------ Functions ------------------
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ------------------ Configuration ------------------
START_YEAR=${START_YEAR:-1960}
END_YEAR=${END_YEAR:-1969}
CHUNK_SIZE=${CHUNK_SIZE:-2}  # Process 2 years at a time

# ------------------ Paths ------------------
SRC_PATH="/pscratch/sd/o/olawale/E3SM_data/v3.LR.historical_0101_bonus.eam.h1_h2/archive/atm/hist"
DES_PATH="/pscratch/sd/o/olawale/E3SM_data/data_processing/regridded_data"
MAP="/global/cfs/cdirs/e3sm/mahf708/misc/map_ne30pg2_to_gaussian_180x360_latlon.nc"
file_types=("h0" "h1" "h2")

# ------------------ Validation ------------------
[[ -d "$SRC_PATH" ]] || { log_msg "❌ Source path not found: $SRC_PATH"; exit 1; }
[[ -f "$MAP" ]] || { log_msg "❌ Map file not found: $MAP"; exit 1; }
command -v ncremap >/dev/null || { log_msg "❌ ncremap not found in PATH"; exit 1; }

mkdir -p "${DES_PATH}"

# ------------------ Global counters ------------------
total_input_count=0
total_regridded_count=0
shopt -s nullglob
start_time=$(date +%s)

log_msg "============================================"
log_msg "Starting ncremap E3SM regridding job"
log_msg "Nodes: $SLURM_NNODES  Tasks: $SLURM_NTASKS"
log_msg "Year range: $START_YEAR-$END_YEAR (chunk size: $CHUNK_SIZE)"
log_msg "============================================"

# ------------------ Main Processing Loop ------------------
for ((start=START_YEAR; start<=END_YEAR; start+=CHUNK_SIZE)); do
    end=$((start + CHUNK_SIZE - 1))
    [[ $end -gt $END_YEAR ]] && end=$END_YEAR
    YEARS=($(seq $start $end))

    log_msg "🚀 Processing years: ${YEARS[*]}"

    # Collect files
    FILES=()
    for file_type in "${file_types[@]}"; do
        for y in "${YEARS[@]}"; do
            FILES+=("${SRC_PATH}"/*eam*${file_type}.*${y}*.nc)
        done
    done
    
    expected_count=${#FILES[@]}
    (( total_input_count += expected_count ))

    if (( expected_count == 0 )); then
        log_msg "⚠️  No files found for ${YEARS[*]}; skipping."
        continue
    fi

    log_msg "🔍 Found $expected_count input files"

    # Run ncremap with error checking
    if ! ncremap --par_typ=mpi \
                  --job_nbr=${SLURM_NTASKS} \
                  --thr_nbr=$OMP_NUM_THREADS \
                  -m "${MAP}" \
                  -O "${DES_PATH}" \
                  -j $expected_count \
                  "${FILES[@]}"; then
        log_msg "❌ ncremap failed for years ${YEARS[*]}"
        exit 1
    fi

    # Count regridded files properly
    regridded_count=0
    for file_type in "${file_types[@]}"; do
        for y in "${YEARS[@]}"; do
            count=$(ls "${DES_PATH}"/*eam*${file_type}.*${y}*.nc 2>/dev/null | wc -l)
            (( regridded_count += count ))
        done
    done

    (( total_regridded_count += regridded_count ))

    # Per-chunk summary
    log_msg "📊 Chunk summary (${YEARS[*]}):"
    log_msg "   Input:     $expected_count"
    log_msg "   Regridded: $regridded_count"

    if [[ $expected_count -ne $regridded_count ]]; then
        log_msg "❌ Mismatch in chunk ${YEARS[*]}"
        exit 1
    fi

    log_msg "============================================"
done

# ------------------ Final Summary ------------------
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

log_msg "============================================"
log_msg "📈 FINAL GLOBAL SUMMARY"
log_msg "🔹 Total input files:     $total_input_count"
log_msg "🔹 Total regridded files: $total_regridded_count"

if [[ $total_input_count -eq $total_regridded_count ]]; then
    log_msg "✅ SUCCESS: All files regridded successfully"
else
    log_msg "❌ FAILURE: Count mismatch!"
    exit 1
fi

log_msg "⏱️  Elapsed time: $((elapsed/3600))h $((elapsed%3600/60))m $((elapsed%60))s"
log_msg "📁 Output directory: ${DES_PATH}"
log_msg "============================================"