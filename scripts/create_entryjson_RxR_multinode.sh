#!/bin/bash

set -e
set -o pipefail

# ---===--- 1. Set up the Python environment ---===---

# Detect cluster based on terminal prompt or hostname
if [[ "$PS1" == *"rorqual"* ]] || [[ "$HOSTNAME" == *"rorqual"* ]] || [[ "$PS1" == *"rc"* ]] || [[ "$HOSTNAME" == *"rc"* ]]; then
    CLUSTER="RORQUAL"
elif [[ "$PS1" == *"tri"* ]] || [[ "$HOSTNAME" == *"tri"* ]]; then
    CLUSTER="TRILLIUM"
elif [[ "$PS1" == *"klogin"* ]] || [[ "$HOSTNAME" == *"klogin"* ]] || [[ "$PS1" == *"kn"* ]] || [[ "$HOSTNAME" == *"kn"* ]]; then
    CLUSTER="KILLARNEY"
elif [[ "$PS1" == *"narval"* ]] || [[ "$HOSTNAME" == *"narval"* ]] || [[ "$PS1" == *"nc"* ]] || [[ "$HOSTNAME" == *"nc"* ]]; then
    CLUSTER="NARVAL"
elif [[ "$PS1" == *"login"* ]] || [[ "$HOSTNAME" == *"login"* ]] || [[ "$PS1" == *"f"* ]] || [[ "$HOSTNAME" == *"f"* ]]; then
    CLUSTER="FIR"
else
    echo "Warning: Could not detect cluster from PS1 or HOSTNAME. Defaulting to RORQUAL."
    CLUSTER="RORQUAL"
fi
echo "Detected cluster: $CLUSTER"

NODE_COUNT="${SLURM_NNODES:-1}"
NODE_INDEX="${SLURM_PROCID:-0}"
INPUT_DATASET_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dataset)
            INPUT_DATASET_DIR="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            exit 1
            ;;
    esac
done

if [[ -z "$INPUT_DATASET_DIR" ]]; then
    echo "Error: --input_dataset is required."
    exit 1
fi

if [[ "$PWD" == *RxR* ]]; then
    PROJECT_DIR="${PWD%%RxR*}/RxR"
else
    echo "Error: Could not find 'RxR' in the current path."
    exit 1
fi
SYSCONFIG_DIR_PATH="$PROJECT_DIR/scripts"
export PYTHONPATH="$PYTHONPATH:$SYSCONFIG_DIR_PATH"

# Load modules strictly before virtualenv activation
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow

if [[ "$CLUSTER" == "TRILLIUM" ]]; then
    export VENV_RXR="/home/indrisch/venv_rxr/"
    if [[ ! -d "${VENV_RXR}" ]]; then
        virtualenv --no-download "${VENV_RXR}"
    fi
    source ${VENV_RXR}/bin/activate
    pip install --no-index --upgrade pip setuptools wheel || true
    pip install --no-index numpy torch pyarrow h5py opencv-python huggingface_hub tqdm pillow || true
else
    if [[ -z "$SLURM_TMPDIR" ]]; then
        export VENV_RXR="/scratch/indrisch/venv_rxr/" 
        if [[ ! -d "${VENV_RXR}" ]]; then
            virtualenv --no-download "${VENV_RXR}"
        fi
        source ${VENV_RXR}/bin/activate
        pip install --no-index --upgrade pip setuptools wheel || true
        pip install --no-index numpy torch pyarrow h5py opencv-python huggingface_hub tqdm pillow || true
    else
        export VENV_RXR="${SLURM_TMPDIR}/venv_rxr/" 
        virtualenv --no-download ${VENV_RXR}
        source ${VENV_RXR}/bin/activate
        pip install --no-index --upgrade pip setuptools wheel
        pip install --no-index numpy torch pyarrow h5py opencv-python huggingface_hub tqdm pillow
    fi
fi
echo "Venv path: ${VENV_RXR}"

# ---===--- 4. Run Python formatter ---===---

python create_entryjson_RxR_multinode.py \
    --input_dataset "${INPUT_DATASET_DIR}"

