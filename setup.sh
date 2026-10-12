# Recoil calibration environment. fitutils/tfutils moved from narf to wums, so use the
# narf/wums of a WRemnants checkout (override its location with WREM_DIR)
export REC_BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export WREM_DIR=${WREM_DIR:-/work/submit/david_w/WRemnants}
source ${WREM_DIR}/setup.sh

export PYTHONPATH="${REC_BASE}:${REC_BASE}/python:$PYTHONPATH"

export TF_ENABLE_ONEDNN_OPTS="1"
export TF_NUM_INTEROP_THREADS="32"
export TF_NUM_INTRAOP_THREADS="32"
