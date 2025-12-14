# SCREAM v1 Data Generation Example

In developing SCREAM v1, the team made a concerted effort to
generalize and streamline variable output capability to allow
for flexible and hopefully scientifically valuable capabilities.
In this example, we will use SCREAM v1 data to generate all the
data we need to train ACE2 on the fly (during runtime) without
the need for post-processing or outputing expensive 3D variables.
Please refer to [ACE2-ERA5 Training Workflow](ace2-workflow.md) for more
information about ACE2 training.

## Key insights

We use the following capabilities from SCREAM v1:

- binary operations (`plus`, `over`)
- vertical reduction (`vert_avg`)
- conditional sampling (`where`)
- process tendencies (`homme_[x]_tend`)
- io aliasing (`:=`)

All these capabilities are [documented in EAMxx user guide](https://docs.e3sm.org/E3SM/EAMxx/user/diags/).

!!! note ""
    The `over` by constants is not yet in the mainline code. As of this writing, it is only available in the [mahf708:hbc2](https://github.com/mahf708/E3SM/tree/hbc2) branch.

## Specific examples

1. `#!yaml air_temperature_2:=T_mid_where_lev_ge_28_where_lev_lt_46_vert_avg_dp_weighted` defines a new variable `air_temperature_2`. It takes the `T_mid` (temperature at mid-levels) variable, applies a conditional filter to include only levels greater than or equal to 28 and less than 46, and then performs a vertical average weighted by pressure thickness (`dp`)
2. `#!yaml total_specific_humidity_0:=qc_plus_qv_plus_qi_plus_qr_where_lev_ge_0_where_lev_lt_15_vert_avg_dp_weighted` defines a new variable `total_specific_humidity_0`. It takes the `qc` (cloud water), `qv` (water vapor), `qi` (ice), and `qr` (rain) variables, applies a conditional filter to include only levels greater than or equal to 0 and less than 15, and then performs a vertical average weighted by pressure thickness (`dp`).
3. `#!yaml tendency_of_total_water_path_due_to_advection:=homme_qc_tend_plus_homme_qv_tend_plus_homme_qi_tend_plus_homme_qr_tend_vert_avg_dp_weighted` defines a new variable `tendency_of_total_water_path_due_to_advection`. It takes the `homme_qc_tend` (cloud water tendency), `homme_qv_tend` (water vapor tendency), `homme_qi_tend` (ice tendency), and `homme_qr_tend` (rain tendency) variables, and then performs a vertical average weighted by pressure thickness (`dp`).
4. `#!yaml HGTsfc:=phis_over_gravit` defines a new variable `HGTsfc`. It takes the `phis` (surface geopotential) variable, and dvides by the gravitation acceleration (taken as constant `gravit`) to get surface height.

## Full script

A full script can be found below.

??? example "SCREAM v1 run script"
    ```bash

    #!/bin/bash -fe

    # EAMxx template run script

    main() {

    do_fetch_code=false
    do_create_newcase=true
    do_case_setup=true
    do_case_build=true
    do_case_submit=true

    readonly MACHINE="pm-gpu"
    readonly CHECKOUT="10022025"
    readonly BRANCH="hbc2"
    readonly CHERRY=( )
    readonly COMPILER="gnugpu"
    readonly DEBUG_COMPILE=FALSE
    readonly Q=debug
    readonly QUEUE=${Q}

    # Simulation
    readonly COMPSET="F2010-SCREAMv1"
    readonly RESOLUTION="ne30pg2_ne30pg2"

    readonly CODE_ROOT="/pscratch/sd/m/mahf708/e3sm-repo/test-pr"
    readonly PROJECT="e3sm"

    readonly TUNINGSET="default" # also plus4k and minus4k are options

    readonly CASE_NAME=${RESOLUTION}.${COMPSET}.${CHECKOUT}.${TUNINGSET}

    readonly CASE_ROOT="${SCRATCH}/e3sm_scratch/${MACHINE}/${CASE_NAME}"

    githash_eamxx=`git --git-dir ${CODE_ROOT}/.git rev-parse HEAD`



    # ****************** Lines to be modified ********************************


    # ****************** Lines to be modified ********************************


    # History file frequency (if using default above)
    readonly HIST_OPTION="nmonths"
    readonly HIST_N="1"

    # Run options
    readonly MODEL_START_TYPE="initial"  # "initial", "continue", "branch", "hybrid"
    readonly START_DATE="2000-01-01"     # "" for default, or explicit "0001-01-01"

    # Additional options for 'branch' and 'hybrid'
    readonly GET_REFCASE=false
    readonly RUN_REFDIR=""
    readonly RUN_REFCASE=""
    readonly RUN_REFDATE=""   # same as MODEL_START_DATE for 'branch', can be different for 'hybrid'


    # Sub-directories
    readonly CASE_BUILD_DIR=${CASE_ROOT}/build
    readonly CASE_ARCHIVE_DIR=${CASE_ROOT}/archive

    readonly CASE_SCRIPTS_DIR=${CASE_ROOT}/case_scripts
    readonly CASE_RUN_DIR=${CASE_ROOT}/run

    readonly NUM_PES=4
    if [[ ${MPS_YES} == "TRUE" ]]; then
        readonly PEL_SIM=$(( MPS_NOS * NUM_PES ))
    else
        readonly PEL_SIM=$(( 4 * NUM_PES ))
    fi
    readonly PELAYOUT=${PEL_SIM}"x1"

    readonly WALLTIME="00:25:00"
    readonly STOP_OPTION="nyears"
    readonly STOP_N="10"
    readonly REST_OPTION="nyears"
    readonly REST_N="1"
    readonly RESUBMIT="0"
    readonly DO_SHORT_TERM_ARCHIVING=false

    # Leave empty (unless you understand what it does)
    readonly OLD_EXECUTABLE=""

    # --- Now, do the work ---

    # Make directories created by this script world-readable
    umask 022

    # Fetch code from Github
    fetch_code

    # Create case
    create_newcase

    # Setup
    case_setup

    # Build
    case_build

    # Configure runtime options
    runtime_options

    # Copy script into case_script directory for provenance
    copy_script

    # Submit
    case_submit

    # All done
    echo $'\n----- All done -----\n'

    }

    # =======================
    # Custom user_nl settings
    # =======================

    user_nl() {

        echo "+++ Configuring SCREAM for 128 vertical levels +++"
        ./xmlchange SCREAM_CMAKE_OPTIONS="SCREAM_NP 4 SCREAM_NUM_VERTICAL_LEV 128 SCREAM_NUM_TRACERS 10"

    }

    ######################################################
    ### Most users won't need to change anything below ###
    ######################################################

    #-----------------------------------------------------
    fetch_code() {

        if [ "${do_fetch_code,,}" != "true" ]; then
        echo $'\n----- Skipping fetch_code -----\n'
        return
        fi

        echo $'\n----- Starting fetch_code -----\n'
        local path=${CODE_ROOT}
        local repo=scream

        echo "Cloning $repo repository branch $BRANCH under $path"
        if [ -d "${path}" ]; then
        echo "ERROR: Directory already exists. Not overwriting"
        exit 20
        fi
        mkdir -p ${path}
        pushd ${path}

        # This will put repository, with all code
        git clone git@github.com:E3SM-Project/${repo}.git .

        # Q: DO WE NEED THIS FOR EAMXX?
        # Setup git hooks
        rm -rf .git/hooks
        git clone git@github.com:E3SM-Project/E3SM-Hooks.git .git/hooks
        git config commit.template .git/hooks/commit.template

        # Check out desired branch
        git checkout ${BRANCH}

        # Custom addition
        if [ "${CHERRY}" != "" ]; then
        echo ----- WARNING: adding git cherry-pick -----
        for commit in "${CHERRY[@]}"
        do
            echo ${commit}
            git cherry-pick ${commit}
        done
        echo -------------------------------------------
        fi

        # Bring in all submodule components
        git submodule update --init --recursive

        popd
    }

    #-----------------------------------------------------
    create_newcase() {

        if [ "${do_create_newcase,,}" != "true" ]; then
        echo $'\n----- Skipping create_newcase -----\n'
        return
        fi

        echo $'\n----- Starting create_newcase -----\n'

        # Base arguments
        args=" --case ${CASE_NAME} \
        --output-root ${CASE_ROOT} \
        --script-root ${CASE_SCRIPTS_DIR} \
        --handle-preexisting-dirs u \
        --compset ${COMPSET} \
        --res ${RESOLUTION} \
        --machine ${MACHINE} \
        --compiler ${COMPILER} \
        --walltime ${WALLTIME} \
        --pecount ${PELAYOUT}"

        # Oprional arguments
        if [ ! -z "${PROJECT}" ]; then
        args="${args} --project ${PROJECT}"
        fi
        if [ ! -z "${CASE_GROUP}" ]; then
        args="${args} --case-group ${CASE_GROUP}"
        fi
        if [ ! -z "${QUEUE}" ]; then
        args="${args} --queue ${QUEUE}"
        fi

        ${CODE_ROOT}/cime/scripts/create_newcase ${args}

        if [ $? != 0 ]; then
        echo $'\nNote: if create_newcase failed because sub-directory already exists:'
        echo $'  * delete old case_script sub-directory'
        echo $'  * or set do_newcase=false\n'
        exit 35
        fi

    }

    #-----------------------------------------------------
    case_setup() {

        if [ "${do_case_setup,,}" != "true" ]; then
        echo $'\n----- Skipping case_setup -----\n'
        return
        fi

        echo $'\n----- Starting case_setup -----\n'
        pushd ${CASE_SCRIPTS_DIR}

        # Setup some CIME directories
        ./xmlchange EXEROOT=${CASE_BUILD_DIR}
        ./xmlchange RUNDIR=${CASE_RUN_DIR}

        # Short term archiving
        ./xmlchange DOUT_S=${DO_SHORT_TERM_ARCHIVING}
        ./xmlchange DOUT_S_ROOT=${CASE_ARCHIVE_DIR}

        # Extracts input_data_dir in case it is needed for user edits to the namelist later
        local input_data_dir=`./xmlquery DIN_LOC_ROOT --value`

        # Custom user_nl
        user_nl

        MPS_NUMBER=4

        ./xmlchange --file env_mach_pes.xml NTHRDS="1"
        ./xmlchange --file env_mach_pes.xml NTHRDS_ATM="1"
        ./xmlchange --file env_mach_pes.xml NTHRDS_LND="$(( 64 / MPS_NUMBER ))"
        ./xmlquery NTHRDS_LND
        ./xmlchange --file env_mach_pes.xml NTHRDS_ICE="$(( 64 / MPS_NUMBER ))"
        ./xmlquery NTHRDS_ICE
        ./xmlchange --file env_mach_pes.xml NTHRDS_OCN="1"
        ./xmlchange --file env_mach_pes.xml NTHRDS_ROF="1"
        ./xmlchange --file env_mach_pes.xml NTHRDS_CPL="1"
        ./xmlchange --file env_mach_pes.xml NTHRDS_GLC="1"
        ./xmlchange --file env_mach_pes.xml NTHRDS_WAV="1"


        ./xmlchange PIO_NETCDF_FORMAT="64bit_data"

        # Finally, run CIME case.setup
        ./case.setup --reset

        # Save provenance invfo
        echo "branch hash for EAMxx: $githash_eamxx" > GIT_INFO.txt
        echo "master hash for output files: $githash_screamdocs" >> GIT_INFO.txt

        popd
    }

    #-----------------------------------------------------
    case_build() {

        pushd ${CASE_SCRIPTS_DIR}

        # do_case_build = false
        if [ "${do_case_build,,}" != "true" ]; then

        echo $'\n----- case_build -----\n'

        if [ "${OLD_EXECUTABLE}" == "" ]; then
            # Ues previously built executable, make sure it exists
            if [ -x ${CASE_BUILD_DIR}/e3sm.exe ]; then
            echo 'Skipping build because $do_case_build = '${do_case_build}
            else
            echo 'ERROR: $do_case_build = '${do_case_build}' but no executable exists for this case.'
            exit 297
            fi
        else
            # If absolute pathname exists and is executable, reuse pre-exiting executable
            if [ -x ${OLD_EXECUTABLE} ]; then
            echo 'Using $OLD_EXECUTABLE = '${OLD_EXECUTABLE}
            cp -fp ${OLD_EXECUTABLE} ${CASE_BUILD_DIR}/
            else
            echo 'ERROR: $OLD_EXECUTABLE = '$OLD_EXECUTABLE' does not exist or is not an executable file.'
            exit 297
            fi
        fi
        echo 'WARNING: Setting BUILD_COMPLETE = TRUE.  This is a little risky, but trusting the user.'
        ./xmlchange BUILD_COMPLETE=TRUE

        # do_case_build = true
        else

        echo $'\n----- Starting case_build -----\n'

        # Turn on debug compilation option if requested
        if [ "${DEBUG_COMPILE}" == "TRUE" ]; then
            ./xmlchange DEBUG=${DEBUG_COMPILE}
        fi

        # Run CIME case.build
        ./case.build

        # Some user_nl settings won't be updated to *_in files under the run directory
        # Call preview_namelists to make sure *_in and user_nl files are consistent.
        ./preview_namelists

        fi

        popd
    }

    #-----------------------------------------------------
    runtime_options() {

        echo $'\n----- Starting runtime_options -----\n'
        pushd ${CASE_SCRIPTS_DIR}

        local input_data_dir=`./xmlquery DIN_LOC_ROOT --value`

        # Set simulation start date
        if [ ! -z "${START_DATE}" ]; then
        ./xmlchange RUN_STARTDATE=${START_DATE}
        fi



        ./atmchange homme::compute_tendencies=qr,qv,qi,qc
        ./atmchange eamxx::compute_tendencies=qr,qv,qi,qc


    cat << EOF >> user_nl_elm

    hist_dov2xy = .true.,.true.
    hist_fincl2 = 'H2OSNO','SOILWATER_10CM','TG'
    hist_mfilt = 1,120
    hist_nhtfrq = 0,-24
    hist_avgflag_pertape = 'A','A'

    EOF

    cat << EOF >> user_nl_cpl
    ocn_surface_flux_scheme = 2
    EOF

        # Segment length
        ./xmlchange STOP_OPTION=${STOP_OPTION,,},STOP_N=${STOP_N}

        # Restart frequency
        ./xmlchange REST_OPTION=${REST_OPTION,,},REST_N=${REST_N}

        # Coupler history
        ./xmlchange HIST_OPTION=${HIST_OPTION,,},HIST_N=${HIST_N}

        # Coupler budgets (always on)
        ./xmlchange BUDGETS=TRUE

        # Set resubmissions
        if (( RESUBMIT > 0 )); then
        ./xmlchange RESUBMIT=${RESUBMIT}
        fi

        # Run type
        # Start from default of user-specified initial conditions
        if [ "${MODEL_START_TYPE,,}" == "initial" ]; then
        ./xmlchange RUN_TYPE="startup"
        ./xmlchange CONTINUE_RUN="FALSE"

        # Continue existing run
        elif [ "${MODEL_START_TYPE,,}" == "continue" ]; then
        ./xmlchange CONTINUE_RUN="TRUE"

        elif [ "${MODEL_START_TYPE,,}" == "branch" ] || [ "${MODEL_START_TYPE,,}" == "hybrid" ]; then
        ./xmlchange RUN_TYPE=${MODEL_START_TYPE,,}
        ./xmlchange GET_REFCASE=${GET_REFCASE}
        ./xmlchange RUN_REFDIR=${RUN_REFDIR}
        ./xmlchange RUN_REFCASE=${RUN_REFCASE}
        ./xmlchange RUN_REFDATE=${RUN_REFDATE}
        echo 'Warning: $MODEL_START_TYPE = '${MODEL_START_TYPE}
        echo '$RUN_REFDIR = '${RUN_REFDIR}
        echo '$RUN_REFCASE = '${RUN_REFCASE}
        echo '$RUN_REFDATE = '${START_DATE}

        else
        echo 'ERROR: $MODEL_START_TYPE = '${MODEL_START_TYPE}' is unrecognized. Exiting.'
        exit 380
        fi

    cat << EOF >> 6hi.yaml
    averaging_type: instant
    fields:
    physics_pg2:
        field_names:
        ####
        - air_temperature_0:=T_mid_where_lev_ge_0_where_lev_lt_15_vert_avg_dp_weighted
        - air_temperature_1:=T_mid_where_lev_ge_15_where_lev_lt_28_vert_avg_dp_weighted
        - air_temperature_2:=T_mid_where_lev_ge_28_where_lev_lt_46_vert_avg_dp_weighted
        - air_temperature_3:=T_mid_where_lev_ge_46_where_lev_lt_62_vert_avg_dp_weighted
        - air_temperature_4:=T_mid_where_lev_ge_62_where_lev_lt_78_vert_avg_dp_weighted
        - air_temperature_5:=T_mid_where_lev_ge_78_where_lev_lt_96_vert_avg_dp_weighted
        - air_temperature_6:=T_mid_where_lev_ge_96_where_lev_lt_115_vert_avg_dp_weighted
        - air_temperature_7:=T_mid_where_lev_ge_115_where_lev_le_128_vert_avg_dp_weighted
        ####
        - eastward_wind_0:=U_where_lev_ge_0_where_lev_lt_15_vert_avg_dp_weighted
        - eastward_wind_1:=U_where_lev_ge_15_where_lev_lt_28_vert_avg_dp_weighted
        - eastward_wind_2:=U_where_lev_ge_28_where_lev_lt_46_vert_avg_dp_weighted
        - eastward_wind_3:=U_where_lev_ge_46_where_lev_lt_62_vert_avg_dp_weighted
        - eastward_wind_4:=U_where_lev_ge_62_where_lev_lt_78_vert_avg_dp_weighted
        - eastward_wind_5:=U_where_lev_ge_78_where_lev_lt_96_vert_avg_dp_weighted
        - eastward_wind_6:=U_where_lev_ge_96_where_lev_lt_115_vert_avg_dp_weighted
        - eastward_wind_7:=U_where_lev_ge_115_where_lev_le_128_vert_avg_dp_weighted
        ####
        - northward_wind_0:=V_where_lev_ge_0_where_lev_lt_15_vert_avg_dp_weighted
        - northward_wind_1:=V_where_lev_ge_15_where_lev_lt_28_vert_avg_dp_weighted
        - northward_wind_2:=V_where_lev_ge_28_where_lev_lt_46_vert_avg_dp_weighted
        - northward_wind_3:=V_where_lev_ge_46_where_lev_lt_62_vert_avg_dp_weighted
        - northward_wind_4:=V_where_lev_ge_62_where_lev_lt_78_vert_avg_dp_weighted
        - northward_wind_5:=V_where_lev_ge_78_where_lev_lt_96_vert_avg_dp_weighted
        - northward_wind_6:=V_where_lev_ge_96_where_lev_lt_115_vert_avg_dp_weighted
        - northward_wind_7:=V_where_lev_ge_115_where_lev_le_128_vert_avg_dp_weighted
        ####
        - total_specific_humidity_0:=qc_plus_qv_plus_qi_plus_qr_where_lev_ge_0_where_lev_lt_15_vert_avg_dp_weighted
        - total_specific_humidity_1:=qc_plus_qv_plus_qi_plus_qr_where_lev_ge_15_where_lev_lt_28_vert_avg_dp_weighted
        - total_specific_humidity_2:=qc_plus_qv_plus_qi_plus_qr_where_lev_ge_28_where_lev_lt_46_vert_avg_dp_weighted
        - total_specific_humidity_3:=qc_plus_qv_plus_qi_plus_qr_where_lev_ge_46_where_lev_lt_62_vert_avg_dp_weighted
        - total_specific_humidity_4:=qc_plus_qv_plus_qi_plus_qr_where_lev_ge_62_where_lev_lt_78_vert_avg_dp_weighted
        - total_specific_humidity_5:=qc_plus_qv_plus_qi_plus_qr_where_lev_ge_78_where_lev_lt_96_vert_avg_dp_weighted
        - total_specific_humidity_6:=qc_plus_qv_plus_qi_plus_qr_where_lev_ge_96_where_lev_lt_115_vert_avg_dp_weighted
        - total_specific_humidity_7:=qc_plus_qv_plus_qi_plus_qr_where_lev_ge_115_where_lev_le_128_vert_avg_dp_weighted
        ####
        - PRESsfc:=ps
        - surface_temperature:=surf_radiative_T
        ####
        - land_fraction:=landfrac
        - ocean_fraction:=ocnfrac
        - sea_ice_fraction:=icefrac
        ####
        - tendency_of_total_water_path_due_to_advection:=homme_qc_tend_plus_homme_qv_tend_plus_homme_qi_tend_plus_homme_qr_tend_vert_avg_dp_weighted
        ####
    max_snapshots_per_file: 4
    filename_prefix: 6hi
    iotype: pnetcdf
    output_control:
    frequency: 6
    frequency_units: nhours
    restart:
    force_new_file: true
    EOF

    cat << EOF >> 6ha.yaml
    averaging_type: average
    fields:
    physics_pg2:
        field_names:
        ####
        - DSWRFtoa:=SW_flux_dn_at_model_top
        - DSWRFsfc:=SW_flux_dn_at_model_bot
        ####
        - DLWRFtoa:=LW_flux_dn_at_model_top
        - DLWRFsfc:=LW_flux_dn_at_model_bot
        ####
        - ULWRFtoa:=LW_flux_up_at_model_top
        - ULWRFsfc:=LW_flux_up_at_model_bot
        ####
        - USWRFtoa:=SW_flux_up_at_model_top
        - USWRFsfc:=SW_flux_up_at_model_bot
        ####
        - LHTFLsfc:=surface_upward_latent_heat_flux
        - SHTFLsfc:=surf_sens_flux
        ####
        - PRATEsfc:=precip_total_surf_mass_flux_over_rho_h2o
        ####
        - HGTsfc:=phis_over_gravit
        ####
    max_snapshots_per_file: 4
    filename_prefix: 6ha
    iotype: pnetcdf
    output_control:
    frequency: 6
    frequency_units: nhours
    restart:
    force_new_file: true
    EOF

        ./atmchange output_yaml_files="./6hi.yaml"
        ./atmchange output_yaml_files+="./6ha.yaml"

        if [[ ${TUNINGSET} == "default" ]]; then
        echo "--------------------------------------------------------------------"
        echo "---------------------USING default SST------------------------------"
        echo "--------------------------------------------------------------------"
        ./xmlchange --file env_run.xml --id SSTICE_DATA_FILENAME --val "${input_data_dir}/ocn/docn7/SSTDATA/sst_ice_CMIP6_DECK_E3SM_1x1_2010_clim_c20220426.nc"
        elif [[ ${TUNINGSET} == "plus1k" ]]; then
        echo "--------------------------------------------------------------------"
        echo "---------------------USING P1K SST----------------------------------"
        echo "--------------------------------------------------------------------"
        ./xmlchange --file env_run.xml --id SSTICE_DATA_FILENAME --val "${input_data_dir}/ocn/docn7/SSTDATA/sst_ice_CMIP6_DECK_E3SM_1x1_2010_clim_plus1k_c20220426.nc"
        elif [[ ${TUNINGSET} == "minus1k" ]]; then
        echo "--------------------------------------------------------------------"
        echo "---------------------USING M1K SST----------------------------------"
        echo "--------------------------------------------------------------------"
        ./xmlchange --file env_run.xml --id SSTICE_DATA_FILENAME --val "${input_data_dir}/ocn/docn7/SSTDATA/sst_ice_CMIP6_DECK_E3SM_1x1_2010_clim_minus1k_c20220426.nc"
        fi

        popd
    }

    #-----------------------------------------------------
    case_submit() {

        if [ "${do_case_submit,,}" != "true" ]; then
        echo $'\n----- Skipping case_submit -----\n'
        return
        fi

        echo $'\n----- Starting case_submit -----\n'
        pushd ${CASE_SCRIPTS_DIR}

        # Run CIME case.submit
        ./case.submit -a="--mail-type=ALL --mail-user=$USER@nersc.gov"
        #./case.submit -a="--qos=${Q}"

        popd
    }

    #-----------------------------------------------------
    copy_script() {

        echo $'\n----- Saving run script for provenance -----\n'

        local script_provenance_dir=${CASE_SCRIPTS_DIR}/run_script_provenance
        mkdir -p ${script_provenance_dir}
        local this_script_name=$( basename -- "$0"; )
        local this_script_dir=$( dirname -- "$0"; )
        local script_provenance_name=${this_script_name}.`date +%Y%m%d-%H%M%S`
        cp -vp "${this_script_dir}/${this_script_name}" ${script_provenance_dir}/${script_provenance_name}

    }

    #-----------------------------------------------------
    # Silent versions of popd and pushd
    pushd() {
        command pushd "$@" > /dev/null
    }
    popd() {
        command popd "$@" > /dev/null
    }

    # Now, actually run the script
    #-----------------------------------------------------
    main
    ```
