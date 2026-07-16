#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-02:00:00
#SBATCH --mem=0
#SBATCH --output=out/%N-format_RxR_multinode-%j.out
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

INPUT_TAR_GZ="/scratch/indrisch/RxR.tar.gz"
INPUT_DATASET_DIR="/scratch/indrisch/RxR_data/"

if [[ -d "${INPUT_DATASET_DIR}" ]]; then
	MODE="--input-dataset-dir"
	INPUT="${INPUT_DATASET_DIR}"
else
	MODE="--input-tar-gz"
	INPUT="${INPUT_TAR_GZ}"
fi

if [[ "$CLUSTER" == "NARVAL" ]]; then
	BASE_PATH="/project/def-wangcs/indrisch/RxR/scripts/"
else
	BASE_PATH="/scratch/indrisch/RxR/scripts/"
fi

srun \
	--ntasks="${SLURM_NNODES:-1}" \
	--ntasks-per-node=1 \
	--cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
	env SPAR7M_SKIP_FINAL_PACKAGING=0 \
	${BASE_PATH}/format_RxR_multinode.sh \
	${MODE} "${INPUT}"

FINAL_DATASET_DIR="/scratch/indrisch/RxR_data_combined_h5_multinode"
FINAL_DATASET_TAR_GZ="/scratch/indrisch/RxR_data_combined_h5_multinode.tar.gz"
if [[ -e "${FINAL_DATASET_TAR_GZ}" ]]; then
	echo "Error: Final tar archive already exists: ${FINAL_DATASET_TAR_GZ}" >&2
	echo "Remove or rename it before rerunning to avoid overwriting a completed run." >&2
fi

echo "file count in FINAL_DATASET_DIR: $(find "${FINAL_DATASET_DIR}" -type f | wc -l)"
echo "disk usage of FINAL_DATASET_DIR: $(du -sh "${FINAL_DATASET_DIR}")"