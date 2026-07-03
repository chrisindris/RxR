#!/bin/bash
# Run this in an interactive job:
: << 'EOF'
salloc \
--time=0-00:05:00 \
--cpus-per-task=1 \
--mem=2G \
--ntasks=1 \
--mail-user=christopher.indris@torontomu.ca \
--mail-type=ALL
EOF
# Interactive job notes:
# process_count=64 and thread_count=16 are highest you can go without freezing
# for many small files, they recommend process_count=12 at most

pushd "/scratch/indrisch/RxR_data/" || exit 1

gcloud auth login --no-launch-browser
gcloud config set storage/process_count 12
gcloud config set storage/thread_count 4

gcloud storage cp --recursive --no-clobber gs://rxr-data .

popd || exit 0