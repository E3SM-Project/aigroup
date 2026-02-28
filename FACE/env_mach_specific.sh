# This file is for user convenience only and is not used by the model
# Changes to this file will be ignored and overwritten
# Changes to the environment should be made in env_mach_specific.xml
# Run ./case.setup --reset to regenerate this file
. /usr/share/lmod/8.3.1/init/sh
module unload cpe cray-hdf5-parallel cray-netcdf-hdf5parallel cray-parallel-netcdf cray-netcdf cray-hdf5 pytorch PrgEnv-gnu PrgEnv-intel PrgEnv-nvidia PrgEnv-cray PrgEnv-aocc gcc-native intel intel-oneapi nvidia aocc cudatoolkit climate-utils cray-libsci matlab craype-accel-nvidia80 craype-accel-host perftools-base perftools darshan
module load PrgEnv-gnu/8.5.0 gcc-native/12.3 cray-libsci/24.07.0 pytorch/2.8.0 craype-accel-host craype/2.7.32 cray-mpich/8.1.30 cray-hdf5-parallel/1.12.2.9 cray-netcdf-hdf5parallel/4.9.0.9 cray-parallel-netcdf/1.12.3.13 cmake/3.30.2
export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_MPIIO_DVS_MAXNODES=1
export HDF5_USE_FILE_LOCKING=FALSE
export PERL5LIB=/global/cfs/cdirs/e3sm/perl/lib/perl5-only-switch
export FI_MR_CACHE_MONITOR=kdreg2
export MPICH_COLL_SYNC=MPI_Bcast
export NETCDF_PATH=/opt/cray/pe/netcdf-hdf5parallel/4.9.0.9/gnu/12.3
export PNETCDF_PATH=/opt/cray/pe/parallel-netcdf/1.12.3.13/gnu/12.3
export GATOR_INITIAL_MB=4000MB
export LD_LIBRARY_PATH=/opt/cray/pe/parallel-netcdf/1.12.3.13/gnu/12.3/lib:/opt/cray/pe/netcdf-hdf5parallel/4.9.0.9/gnu/12.3/lib:/opt/cray/pe/hdf5-parallel/1.12.2.9/gnu/12.3/lib:/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/lib:/opt/cray/pe/mpich/8.1.30/gtl/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/lib64:/opt/cray/pe/libsci/24.07.0/GNU/12.3/x86_64/lib:/opt/cray/pe/dsmml/0.3.1/dsmml/lib:/opt/cray/pe/parallel-netcdf/1.12.3.13/gnu/12.3/lib:/opt/cray/pe/netcdf-hdf5parallel/4.9.0.9/gnu/12.3/lib:/opt/cray/pe/hdf5-parallel/1.12.2.9/gnu/12.3/lib:/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/lib:/opt/cray/pe/mpich/8.1.30/gtl/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/lib64:/opt/cray/pe/libsci/24.07.0/GNU/12.3/x86_64/lib:/opt/cray/pe/dsmml/0.3.1/dsmml/lib:/global/common/software/nersc9/nccl/2.24.3/plugin/lib:/global/common/software/nersc9/nccl/2.24.3/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/extras/CUPTI/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/extras/Debugger/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/nvvm/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/lib64:/opt/cray/pe/parallel-netcdf/1.12.3.13/gnu/12.3/lib:/opt/cray/pe/netcdf-hdf5parallel/4.9.0.9/gnu/12.3/lib:/opt/cray/pe/hdf5-parallel/1.12.2.9/gnu/12.3/lib:/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/lib:/opt/cray/pe/mpich/8.1.30/gtl/lib:/opt/cray/pe/libsci/24.07.0/GNU/12.3/x86_64/lib:/opt/cray/pe/dsmml/0.3.1/dsmml/lib:/opt/cray/libfabric/1.22.0/lib64
export LD_LIBRARY_PATH=/global/common/software/nersc9/pytorch/2.8.0/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${FTorch_ROOT}/lib64/ 

export MPICH_SMP_SINGLE_COPY_MODE=CMA
export PKG_CONFIG_PATH=/global/cfs/cdirs/e3sm/3rdparty/protobuf/21.6/gcc-native-12.3/lib/pkgconfig:/global/cfs/cdirs/e3sm/3rdparty/protobuf/21.6/gcc-native-12.3/lib/pkgconfig:/opt/cray/pe/parallel-netcdf/1.12.3.13/gnu/12.3/lib/pkgconfig:/opt/cray/pe/netcdf-hdf5parallel/4.9.0.9/gnu/12.3/lib/pkgconfig:/opt/cray/pe/hdf5-parallel/1.12.2.9/gnu/12.3/lib/pkgconfig:/opt/cray/pe/craype/2.7.32/pkg-config:/opt/cray/pe/modulefiles/cudatoolkit:/opt/cray/pe/dsmml/0.3.1/dsmml/lib/pkgconfig:/global/cfs/cdirs/e3sm/3rdparty/protobuf/21.6/gcc-native-12.3/lib/pkgconfig:/opt/cray/libfabric/1.22.0/lib64/pkgconfig
export ADIOS2_ROOT=/global/cfs/cdirs/e3sm/3rdparty/adios2/2.10.2/cray-mpich-8.1.28/gcc-native-12.3
export BLOSC2_ROOT=/global/cfs/cdirs/e3sm/3rdparty/c-blosc2/2.15.2/gcc-native-12.3
export MGARD_ROOT=/global/cfs/cdirs/e3sm/3rdparty/mgard/1.5.2/gcc-native-12.3
export SZ_ROOT=/global/cfs/cdirs/e3sm/3rdparty/sz/2.1.12.5/gcc-native-12.3
export ZFP_ROOT=/global/cfs/cdirs/e3sm/3rdparty/zfp/1.0.1/gcc-native-12.3
export BLA_VENDOR=Generic
export Albany_ROOT=/global/common/software/e3sm/albany/2024.03.26/gcc/11.2.0
export Trilinos_ROOT=/global/common/software/e3sm/trilinos/15.1.1/gcc/11.2.0
export MOAB_ROOT=/global/cfs/cdirs/e3sm/software/moab/gnu
export FTorch_ROOT=/global/cfs/cdirs/e3sm/anolan/FTorch_v1.0.0-pytorch_v2.8.0
ulimit -s unlimited
