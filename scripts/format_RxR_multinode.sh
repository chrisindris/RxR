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
INPUT_TAR_GZ=""
INPUT_DATASET_DIR=""
FINAL_DATASET_DIR="/scratch/indrisch/RxR_data_combined_h5_multinode"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-tar-gz)
            INPUT_TAR_GZ="$2"
            shift 2
            ;;
        --input-dataset-dir)
            INPUT_DATASET_DIR="$2"
            shift 2
            ;;
        --final-dataset-dir)
            FINAL_DATASET_DIR="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            exit 1
            ;;
    esac
done

if [[ -z "$INPUT_TAR_GZ" && -z "$INPUT_DATASET_DIR" ]]; then
    echo "Error: One of --input-tar-gz or --input-dataset-dir is required."
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

# Load modules
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow

if [[ "$CLUSTER" == "TRILLIUM" ]]; then
    export VENV_RXR="/home/indrisch/venv_rxr/"
    source ${VENV_RXR}/bin/activate
else
    if [[ -z "$SLURM_TMPDIR" ]]; then
        export VENV_RXR="/scratch/indrisch/venv_rxr/" 
        source ${VENV_RXR}/bin/activate
    else
        export VENV_RXR="${SLURM_TMPDIR}/venv_rxr/" 
        virtualenv --no-download ${VENV_RXR}
        source ${VENV_RXR}/bin/activate
        pip install --no-index --upgrade pip setuptools wheel
        pip install --no-index numpy torch pyarrow h5py opencv-python huggingface_hub tqdm
    fi
fi
echo "Venv path: ${VENV_RXR}"

# ---===--- 2. Extract and assign ZIP files ---===---

if [[ -z "$SLURM_TMPDIR" ]]; then
    export WORKDIR="/scratch/indrisch/RxR_workdir/node_${NODE_INDEX}/"
    export COMBINED_DATASET_DIR="/scratch/indrisch/RxR_data_combined_h5_multinode_temp/"
else
    export WORKDIR="${SLURM_TMPDIR}/RxR_workdir/node_${NODE_INDEX}/"
    export COMBINED_DATASET_DIR="${SLURM_TMPDIR}/RxR_data_combined_h5_multinode_temp/"
fi
mkdir -p "${WORKDIR}"
mkdir -p "${COMBINED_DATASET_DIR}"
echo "WORKDIR: ${WORKDIR}"

# Do not copy the large tarball to SLURM_TMPDIR. Read directly from the source.
TAR_GZ_LOCAL="${INPUT_TAR_GZ}"

echo "Listing contents of ${TAR_GZ_LOCAL} to find zip/7z files..."
# Find all zip/7z files directly from the tarball; however, if we have a file available that simply lists them, use that.
if [[ -f "/project/def-wangcs/indrisch/RxR/secrets/RxR_tar_gz_contents.txt" ]]; then
    ZIP_FILES=($(tail -n +2 "/project/def-wangcs/indrisch/RxR/secrets/RxR_tar_gz_contents.txt" | sort))
else 
    ZIP_FILES=($(tar -tf "${TAR_GZ_LOCAL}" | grep -iE '\.(zip|7z)$' | sort))
fi

# Assign to current node
ASSIGNED_ZIPS=()
for i in "${!ZIP_FILES[@]}"; do
    if (( i % NODE_COUNT == NODE_INDEX )); then
        ASSIGNED_ZIPS+=("${ZIP_FILES[$i]}")
    fi
done

echo "Node $NODE_INDEX assigned ${#ASSIGNED_ZIPS[@]} zip files out of ${#ZIP_FILES[@]} total."

echo "Extracting assigned files from ${TAR_GZ_LOCAL} to ${WORKDIR}/zips"
mkdir -p "${WORKDIR}/zips"
if [ ${#ASSIGNED_ZIPS[@]} -gt 0 ]; then
    tar -xf "${TAR_GZ_LOCAL}" -C "${WORKDIR}/zips" "${ASSIGNED_ZIPS[@]}"
fi

EXTRACT_ROOT="${WORKDIR}/extracted"
mkdir -p "${EXTRACT_ROOT}"

# Find all zip/7z files that were just extracted
shopt -s nullglob
EXTRACTED_ZIPS=($(find "${WORKDIR}/zips" -type f \( -name "*.zip" -o -name "*.7z" \) | sort))

# Extract assigned zip/7z files
for zf in "${EXTRACTED_ZIPS[@]}"; do
    echo "Extracting $zf..."
    if [[ -f ~/7zz ]]; then
        ~/7zz x "$zf" -o"${EXTRACT_ROOT}" -y >/dev/null
        rm -f "$zf"
    else
        echo "Error: ~/7zz not found. Please install 7-zip."
        exit 1
    fi
done

echo "Extraction complete."

# ---===--- 3. Run Python formatter ---===---

python format_Structured3D_multinode.py \
    --combined-dataset "${COMBINED_DATASET_DIR}" \
    --extract-root "${EXTRACT_ROOT}"

# ---===--- 4. Rsync to final destination ---===---

if [[ "${SPAR7M_SKIP_FINAL_PACKAGING:-0}" == "1" ]]; then
    echo "Skipping final packaging in worker step."
else
    echo "Node ${NODE_INDEX}: rsyncing ${COMBINED_DATASET_DIR} to permanent storage ${FINAL_DATASET_DIR}"
    mkdir -p "${FINAL_DATASET_DIR}"
    rsync -auzh --no-p --no-g "${COMBINED_DATASET_DIR%/}/" "${FINAL_DATASET_DIR}/"
    echo "Node ${NODE_INDEX}: rsync complete."
fi
