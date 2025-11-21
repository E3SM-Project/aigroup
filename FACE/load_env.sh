if [ -z ${E3SMU_MACHINE+x} ]; then
    SLURM_JOB_ID="fkdjsl"
    source /global/common/software/e3sm/anaconda_envs/test_e3sm_unified_1.12.0rc4_pm-cpu.sh
fi

module load pytorch/2.6.0

export FTorch_ROOT="/global/cfs/cdirs/e3sm/FTorch"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${FTorch_ROOT}/lib64/
