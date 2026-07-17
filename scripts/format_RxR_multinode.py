#!/usr/bin/env python3

"""Formatter for accumulating RxR dataset pose_traces into stacked HDF5 and JSON index files.

This script scans extracted pose_traces (.npz files) across splits (rxr_train, rxr_val_seen, rxr_val_unseen)
and packs their arbitrary-length matrices into consolidated HDF5 datasets with exact cumulative index bounds
stored in accompanying JSON lookup tables. Supports multi-node sharding for distributed SLURM clusters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(iterable, *args, **kwargs):
		return iterable


SPLIT_MAX_DIMS = {
	"rxr_train": {"k_max": 136, "m_max": 512},
	"rxr_val_seen": {"k_max": 95, "m_max": 512},
	"rxr_val_unseen": {"k_max": 147, "m_max": 512},
}

DEFAULT_MAX_DIMS = {"k_max": 200, "m_max": 512}


def locate_pose_traces_dir(root: Path) -> Path:
	"""Locate the pose_traces base directory starting from root."""
	if (root / "rxr-data" / "pose_traces").is_dir():
		return root / "rxr-data" / "pose_traces"
	if (root / "pose_traces").is_dir():
		return root / "pose_traces"
	if root.name == "pose_traces" and root.is_dir():
		return root
	# Check if root itself contains split directories directly
	for split_name in ("rxr_train", "rxr_val_seen", "rxr_val_unseen"):
		if (root / split_name).is_dir():
			return root
	return root


def ensure_dataset_capacity(
	dataset: h5py.Dataset,
	required_size: int,
	buffer_chunk: int,
) -> None:
	"""Resize dataset along axis 0 if required_size exceeds current capacity."""
	current_capacity = dataset.shape[0]
	if required_size > current_capacity:
		new_capacity = max(required_size, current_capacity + buffer_chunk)
		new_shape = list(dataset.shape)
		new_shape[0] = new_capacity
		dataset.resize(tuple(new_shape))


def trim_dataset(dataset: h5py.Dataset, final_size: int) -> None:
	"""Trim dataset along axis 0 to exactly final_size."""
	if dataset.shape[0] != final_size:
		new_shape = list(dataset.shape)
		new_shape[0] = final_size
		dataset.resize(tuple(new_shape))


def process_split(
	split_dir: Path,
	split_name: Path | str,
	destination_dir: Path,
	node_index: int = 0,
	node_count: int = 1,
	sharding_mode: str = "node",
) -> None:
	split_str = str(split_name)
	max_dims = SPLIT_MAX_DIMS.get(split_str, DEFAULT_MAX_DIMS)
	k_max = max_dims["k_max"]
	m_max = max_dims["m_max"]

	if not split_dir.is_dir():
		print(f"Directory not found for split '{split_str}': {split_dir}. Skipping.")
		return

	all_npz_files = sorted(split_dir.glob("*.npz"))
	if not all_npz_files:
		print(f"No .npz files found in {split_dir}. Skipping.")
		return

	if sharding_mode == "node":
		assigned_files = [
			f for i, f in enumerate(all_npz_files) if i % node_count == node_index
		]
	else:
		# sharding_mode == "split" handled at caller level, process all here
		assigned_files = all_npz_files

	print(
		f"[{split_str}] Node {node_index}/{node_count} processing {len(assigned_files)} of {len(all_npz_files)} files (sharding_mode={sharding_mode})."
	)
	if not assigned_files:
		return

	destination_dir.mkdir(parents=True, exist_ok=True)
	if sharding_mode == "node" and node_count > 1:
		h5_filename = f"node_{node_index}_{split_str}.h5"
		json_filename = f"node_{node_index}_{split_str}.json"
	else:
		h5_filename = f"{split_str}.h5"
		json_filename = f"{split_str}.json"

	h5_path = destination_dir / h5_filename
	json_path = destination_dir / json_filename

	# String dtype for h5py
	str_dtype = (
		h5py.string_dtype(encoding="utf-8")
		if hasattr(h5py, "string_dtype")
		else h5py.special_dtype(vlen=str)
	)

	n_index = 0
	k_index = 0
	x_index = 0

	index_dict: dict[str, object] = {"k_max": k_max, "m_max": m_max}

	# Open HDF5 file
	with h5py.File(h5_path, "w") as h5_file:
		# Initialize datasets with maxshape=(None, ...) along axis 0 for growth
		d_pano = h5_file.create_dataset(
			"pano",
			shape=(0, 1),
			maxshape=(None, 1),
			dtype=str_dtype,
			compression="gzip",
			compression_opts=9,
		)
		d_time = h5_file.create_dataset(
			"time",
			shape=(0, 1),
			maxshape=(None, 1),
			dtype=np.float64,
			compression="gzip",
			compression_opts=9,
		)
		d_audio_time = h5_file.create_dataset(
			"audio_time",
			shape=(0, 1),
			maxshape=(None, 1),
			dtype=np.float64,
			compression="gzip",
			compression_opts=9,
		)
		d_extrinsic = h5_file.create_dataset(
			"extrinsic_matrix",
			shape=(0, 16),
			maxshape=(None, 16),
			dtype=np.float64,
			compression="gzip",
			compression_opts=9,
		)
		d_intrinsic = h5_file.create_dataset(
			"intrinsic_matrix",
			shape=(0, 16),
			maxshape=(None, 16),
			dtype=np.float64,
			compression="gzip",
			compression_opts=9,
		)
		d_image_mask = h5_file.create_dataset(
			"image_mask",
			shape=(0, 128, 256),
			maxshape=(None, 128, 256),
			dtype=np.bool_,
			compression="gzip",
			compression_opts=9,
		)
		d_weights = h5_file.create_dataset(
			"feature_weights",
			shape=(0, 36),
			maxshape=(None, 36),
			dtype=np.float32,
			compression="gzip",
			compression_opts=9,
		)
		d_text_masks = h5_file.create_dataset(
			"text_masks",
			shape=(0, k_max, m_max),
			maxshape=(None, k_max, m_max),
			dtype=np.float32,
			compression="gzip",
			compression_opts=9,
		)

		for npz_path in tqdm(assigned_files, desc=f"Packing {split_str}", unit="file"):
			try:
				with np.load(npz_path) as data:
					pano_arr = data["pano"]
					time_arr = data["time"]
					has_audio_time = "audio_time" in data.files
					if has_audio_time:
						audio_time_arr = data["audio_time"]
					else:
						audio_time_arr = np.zeros_like(time_arr)

					extrinsic_arr = data["extrinsic_matrix"]
					intrinsic_arr = data["intrinsic_matrix"]
					image_mask_arr = data["image_mask"]
					weights_arr = data["feature_weights"]

					has_text_masks = "text_masks" in data.files
					if has_text_masks:
						text_masks_arr = data["text_masks"]
					else:
						text_masks_arr = None

			except (KeyError, EOFError, OSError) as e:
				print(f"Warning: Failed to read {npz_path} ({e}). Skipping file.")
				continue

			n = pano_arr.shape[0]
			k = image_mask_arr.shape[0]
			if has_text_masks and text_masks_arr is not None:
				k_tm, m_tm = text_masks_arr.shape
				m = m_tm
			else:
				m = 0

			# Parse scene ID and trace type
			stem = npz_path.stem  # e.g., 000000_follower_pose_trace
			parts = stem.split("_")
			scene_id = parts[0]
			if "_follower_" in stem:
				trace_type = "follower"
			elif "_guide_" in stem:
				trace_type = "guide"
			else:
				trace_type = parts[1] if len(parts) > 1 else "unknown"

			# Ensure dataset capacity (chunked buffering to minimize resize calls)
			ensure_dataset_capacity(d_pano, n_index + n, buffer_chunk=50000)
			ensure_dataset_capacity(d_time, n_index + n, buffer_chunk=50000)
			ensure_dataset_capacity(d_audio_time, n_index + n, buffer_chunk=50000)
			ensure_dataset_capacity(d_extrinsic, n_index + n, buffer_chunk=50000)
			ensure_dataset_capacity(d_intrinsic, n_index + n, buffer_chunk=50000)
			ensure_dataset_capacity(d_image_mask, k_index + k, buffer_chunk=5000)
			ensure_dataset_capacity(d_weights, k_index + k, buffer_chunk=5000)
			if has_text_masks and text_masks_arr is not None:
				ensure_dataset_capacity(d_text_masks, x_index + 1, buffer_chunk=1000)

			# Write slices
			d_pano[n_index : n_index + n] = pano_arr.reshape(-1, 1)
			d_time[n_index : n_index + n] = time_arr.reshape(-1, 1)
			d_audio_time[n_index : n_index + n] = audio_time_arr.reshape(-1, 1)
			d_extrinsic[n_index : n_index + n] = extrinsic_arr.reshape(n, 16)
			d_intrinsic[n_index : n_index + n] = intrinsic_arr.reshape(n, 16)
			d_image_mask[k_index : k_index + k] = image_mask_arr
			d_weights[k_index : k_index + k] = weights_arr

			if has_text_masks and text_masks_arr is not None:
				padded_tm = np.zeros((k_max, m_max), dtype=text_masks_arr.dtype)
				pad_k = min(k_tm, k_max)
				pad_m = min(m_tm, m_max)
				padded_tm[:pad_k, :pad_m] = text_masks_arr[:pad_k, :pad_m]
				d_text_masks[x_index] = padded_tm
				current_x_index = x_index
				x_index += 1
			else:
				current_x_index = -1

			if scene_id not in index_dict:
				index_dict[scene_id] = {}
			index_dict[scene_id][trace_type] = {
				"n_index": int(n_index),
				"n": int(n),
				"k_index": int(k_index),
				"k": int(k),
				"x_index": int(current_x_index),
				"m": int(m),
				"has_text_masks": bool(has_text_masks),
				"has_audio_time": bool(has_audio_time),
				"file_name": npz_path.name,
			}

			n_index += n
			k_index += k

		# Trim datasets to exact boundaries
		trim_dataset(d_pano, n_index)
		trim_dataset(d_time, n_index)
		trim_dataset(d_audio_time, n_index)
		trim_dataset(d_extrinsic, n_index)
		trim_dataset(d_intrinsic, n_index)
		trim_dataset(d_image_mask, k_index)
		trim_dataset(d_weights, k_index)
		trim_dataset(d_text_masks, x_index)

	# Write JSON index
	with json_path.open("w", encoding="utf-8") as json_file:
		json.dump(index_dict, json_file, indent=2, sort_keys=True)
		json_file.write("\n")

	print(
		f"[{split_str}] Complete. Saved {h5_path} (N={n_index}, K={k_index}, X={x_index}) and {json_path}."
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Pack RxR pose_traces .npz files into stacked HDF5 + JSON indexes."
	)
	parser.add_argument(
		"--extract-root",
		type=Path,
		required=True,
		help="Root directory containing rxr-data/pose_traces or pose_traces.",
	)
	parser.add_argument(
		"--combined-dataset",
		type=Path,
		required=True,
		help="Destination directory for combined HDF5 and JSON files.",
	)
	parser.add_argument(
		"--node-index",
		type=int,
		default=0,
		help="Current node index for SLURM sharding.",
	)
	parser.add_argument(
		"--node-count",
		type=int,
		default=1,
		help="Total node count for SLURM sharding.",
	)
	parser.add_argument(
		"--sharding-mode",
		choices=["node", "split"],
		default="node",
		help="Sharding strategy across nodes: 'node' (shard files within splits) or 'split' (assign splits across nodes).",
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		default=["rxr_train", "rxr_val_seen", "rxr_val_unseen"],
		help="Splits to process.",
	)
	args = parser.parse_args()

	pose_traces_root = locate_pose_traces_dir(args.extract_root)
	print(f"Located pose_traces base directory: {pose_traces_root}")

	for i, split_name in enumerate(args.splits):
		if args.sharding_mode == "split":
			if i % args.node_count != args.node_index:
				print(
					f"Skipping split '{split_name}' on node {args.node_index}/{args.node_count} (sharding_mode=split)."
				)
				continue
		split_dir = pose_traces_root / split_name
		process_split(
			split_dir=split_dir,
			split_name=split_name,
			destination_dir=args.combined_dataset,
			node_index=args.node_index,
			node_count=args.node_count,
			sharding_mode=args.sharding_mode,
		)


if __name__ == "__main__":
	main()
