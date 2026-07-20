#!/usr/bin/env python3

""" Amalgamator script so that we can create a single rxr.json file to use to direct filename queries to the correct existing node_{\d}_rxr_{train|val_seen|val_unseen}.json files.

Since each file "X_follower_pose_trace.npz" already has key X in the dictionary, this is as simple as creating a dictionary with the keys being "train", "val_seen", and "val_unseen", and the value being a dictionary where the keys are the same keys as in the corresponding json files and the values are the corresponding node {\d}. For example, we might index it like d["train"]["X"] = 0, so we will know then to look for "X" in node_0_rxr_train.json.      
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def main():

    parser = argparse.ArgumentParser(
            description="Create a combined rxr.json file for RxR dataset."
            )
    parser.add_argument(
            "--input_dataset",
            type=Path,
            required=True,
            help="Directory for combined HDF5 and JSON files.",
            )
    args = parser.parse_args()
    D = {}

    dataset_dir = args.input_dataset
    jsons = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]

    pattern = r"^node_(\d)_rxr_(train|val_seen|val_unseen)\.json$"

    for f in jsons:
        match = re.match(pattern, f)
        if match:
            node_num = int(match.group(1))
            split = match.group(2)

            with open(os.path.join(dataset_dir, f), 'r') as infile:
                data = json.load(infile)

            if split not in D:
                D[split] = {}

            for key in data.keys():
                D[split][key] = node_num


    output_file = os.path.join(dataset_dir, "rxr.json")
    with open(output_file, 'w') as outfile:
        json.dump(D, outfile, indent=4
                  )

if __name__ == "__main__":
    main()
