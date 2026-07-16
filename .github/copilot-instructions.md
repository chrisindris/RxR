# Copilot Instructions

# General Repository Environment Instructions

We are using AllianceCan (Compute Canada).
You currently have terminal access to the login node, whereas any SLURM script will be run on a compute node.
The login node does not have GPU access, and does not have a SLURM_TMPDIR (temporary directory created only for the SLURM run).
Computer nodes do not have internet access.

# Environment

Though we build and use a temporary venv on the compute nodes when running programs, an identical venv is available on the login nodes for light debugging. This venv can be activated by running the following terminal commands:

```bash
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow
source /scratch/indrisch/venv_RxR/bin/activate
```