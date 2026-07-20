#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:05:00
#SBATCH --mem=0
#SBATCH --output=out/%N-create_entryjson_RxR_multinode-%j.out
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL

set -e
set -o pipefail

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

INPUT_DATASET_DIR="/scratch/indrisch/RxR_data_combined_h5_multinode/"

if [[ -d "${INPUT_DATASET_DIR}" ]]; then
	MODE="--input_dataset"
	INPUT="${INPUT_DATASET_DIR}"
else
	echo "Error: Input dataset directory does not exist: ${INPUT_DATASET_DIR}" >&2
  exit 1
fi

if [[ "$CLUSTER" == "NARVAL" ]]; then
	BASE_PATH="/project/def-wangcs/indrisch/RxR/scripts/"
else
	BASE_PATH="/scratch/indrisch/RxR/scripts/"
fi


${BASE_PATH}/create_entryjson_RxR_multinode.sh \
	${MODE} "${INPUT}"
