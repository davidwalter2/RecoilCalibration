#!/bin/bash
# Run all fits of the DeepMETPVRobust calibration, independent fits in parallel, in the order of
# their dependencies (the data fits need the background and the MC fits), then export the model.
# Each fit uses ~5 cores, run on a large node.
# Run from the repository root after sourcing setup.sh:
#   calibration/run_highPU_DeepMETPVRobust.sh [dataDir] [plotDir] [further options of the driver, e.g. --qtBins ...]

DATADIR=${1:-data/highPU_DeepMETPVRobust_v2}
PLOTDIR=${2:-${HOME}/public_html/WMass/261011_recoilCalibration/highPU_DeepMETPVRobust}
shift $(( $# < 2 ? $# : 2 ))
EXTRA="$@"
LOGDIR=${DATADIR}/logs
mkdir -p ${LOGDIR}

run() {
    python calibration/highPU_DeepMETPVRobust.py --dataDir ${DATADIR} --plotDir ${PLOTDIR} ${EXTRA} --steps "$@" > ${LOGDIR}/$1.log 2>&1
    local rc=$?
    echo "$(date +%T) $1 finished with exit code ${rc}"
}

stage() {
    for step in "$@"; do
        run ${step} &
    done
    wait
}

echo "$(date +%T) MC and background fits"
stage z_para z_perp bkg_para bkg_perp z_para_ext z_perp_ext
echo "$(date +%T) data fits"
stage data_para data_perp data_para_bkgUp data_perp_bkgUp data_para_bkgDown data_perp_bkgDown
# trust-krylov is very slow for the alternative parametrisation, minimize the constrained likelihood directly
run data_para_ext --trustExact &
run data_perp_ext --trustExact &
wait
echo "$(date +%T) export"
run export
