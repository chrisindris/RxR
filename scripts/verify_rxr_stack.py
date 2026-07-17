#!/usr/bin/env python3

"""Verification script for RxR dataset pose_traces HDF5 stacking and JSON indexing.

Performs numerical and structural verification comparing the generated HDF5 slices against
raw .npz files. Includes a self-test suite (--run-self-test) that generates synthetic .npz
data, runs the stacker, and verifies exact numerical reconstruction.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

# Import format_RxR_multinode logic for self-test
try:
	from format_RxR_multinode import process_split, locate_pose_traces_dir
except ImportError:
	sys.path.append(str(Path(__file__).parent))
	from format_RxR_multinode import process_split, locate_pose_traces_dir


def verify_split_shard(
	h5_path: Path,
	json_path: Path,
	raw_split_dir: Path,
) -> tuple[int, int]:
	"""Verify all entries in a JSON index and HDF5 shard against raw .npz files.

	Returns:
		(verified_traces_count, total_snapshots_verified)
	"""
	if not json_path.exists() or not h5_path.exists():
		print(f"Skipping verification: {json_path} or {h5_path} does not exist.")
		return 0, 0

	with json_path.open("r", encoding="utf-8") as f:
		index_dict = json.load(f)

	k_max = index_dict.get("k_max", 200)
	m_max = index_dict.get("m_max", 512)

	verified_traces = 0
	total_snapshots = 0

	with h5py.File(h5_path, "r") as h5_file:
		for scene_id, scene_entry in index_dict.items():
			if scene_id in ("k_max", "m_max"):
				continue
			if not isinstance(scene_entry, dict):
				continue

			for trace_type, trace_meta in scene_entry.items():
				file_name = trace_meta["file_name"]
				raw_npz_path = raw_split_dir / file_name
				if not raw_npz_path.exists():
					print(f"Warning: Raw file not found: {raw_npz_path}")
					continue

				n_index = trace_meta["n_index"]
				n = trace_meta["n"]
				k_index = trace_meta["k_index"]
				k = trace_meta["k"]
				x_index = trace_meta["x_index"]
				m = trace_meta["m"]
				has_text_masks = trace_meta["has_text_masks"]
				has_audio_time = trace_meta.get("has_audio_time", True)

				# Load raw data
				with np.load(raw_npz_path) as raw_data:
					raw_pano = raw_data["pano"]
					raw_time = raw_data["time"]
					raw_audio_time = raw_data["audio_time"] if has_audio_time else np.zeros_like(raw_time)
					raw_extrinsic = raw_data["extrinsic_matrix"]
					raw_intrinsic = raw_data["intrinsic_matrix"]
					raw_image_mask = raw_data["image_mask"]
					raw_weights = raw_data["feature_weights"]
					raw_text_masks = raw_data["text_masks"] if has_text_masks else None

				# Retrieve slices from HDF5
				h5_pano = h5_file["pano"][n_index : n_index + n].flatten()
				h5_time = h5_file["time"][n_index : n_index + n].flatten()
				h5_audio_time = h5_file["audio_time"][n_index : n_index + n].flatten()
				h5_extrinsic = h5_file["extrinsic_matrix"][
					n_index : n_index + n
				].reshape(raw_extrinsic.shape)
				h5_intrinsic = h5_file["intrinsic_matrix"][
					n_index : n_index + n
				].reshape(raw_intrinsic.shape)
				h5_image_mask = h5_file["image_mask"][k_index : k_index + k]
				h5_weights = h5_file["feature_weights"][k_index : k_index + k]

				# Assert equality
				if h5_pano.dtype.kind in ("S", "O"):
					h5_pano_str = np.array(
						[
							s.decode("utf-8") if isinstance(s, bytes) else str(s)
							for s in h5_pano
						],
						dtype=raw_pano.dtype,
					)
					np.testing.assert_array_equal(
						h5_pano_str, raw_pano, err_msg=f"pano mismatch in {file_name}"
					)
				else:
					np.testing.assert_array_equal(
						h5_pano, raw_pano, err_msg=f"pano mismatch in {file_name}"
					)

				np.testing.assert_allclose(
					h5_time, raw_time, err_msg=f"time mismatch in {file_name}"
				)
				if has_audio_time:
					np.testing.assert_allclose(
						h5_audio_time,
						raw_audio_time,
						err_msg=f"audio_time mismatch in {file_name}",
					)
				np.testing.assert_allclose(
					h5_extrinsic,
					raw_extrinsic,
					err_msg=f"extrinsic_matrix mismatch in {file_name}",
				)
				np.testing.assert_allclose(
					h5_intrinsic,
					raw_intrinsic,
					err_msg=f"intrinsic_matrix mismatch in {file_name}",
				)
				np.testing.assert_array_equal(
					h5_image_mask,
					raw_image_mask,
					err_msg=f"image_mask mismatch in {file_name}",
				)
				np.testing.assert_allclose(
					h5_weights,
					raw_weights,
					err_msg=f"feature_weights mismatch in {file_name}",
				)

				if has_text_masks and raw_text_masks is not None:
					k_tm, m_tm = raw_text_masks.shape
					pad_k = min(k_tm, k_max)
					pad_m = min(m_tm, m_max)
					h5_text_mask = h5_file["text_masks"][x_index, :pad_k, :pad_m]
					np.testing.assert_allclose(
						h5_text_mask,
						raw_text_masks[:pad_k, :pad_m],
						err_msg=f"text_masks mismatch in {file_name}",
					)

				verified_traces += 1
				total_snapshots += n

	return verified_traces, total_snapshots


def generate_synthetic_npz(path: Path, n: int, k: int, m: int | None = None) -> None:
	"""Generate a synthetic .npz file matching RxR pose_trace schema."""
	pano = np.array([f"32hexdigitpano{i:018d}" for i in range(n)], dtype="<U32")
	time_arr = np.linspace(0.0, 100.0, n, dtype=np.float64)
	audio_time_arr = np.linspace(0.0, 100.0, n, dtype=np.float64)
	extrinsic = np.random.randn(n, 4, 4).astype(np.float64)
	intrinsic = np.random.randn(n, 4, 4).astype(np.float64)
	image_mask = np.random.rand(k, 128, 256) > 0.5
	feature_weights = np.random.randn(k, 36).astype(np.float32)

	data = {
		"pano": pano,
		"time": time_arr,
		"audio_time": audio_time_arr,
		"extrinsic_matrix": extrinsic,
		"intrinsic_matrix": intrinsic,
		"image_mask": image_mask,
		"feature_weights": feature_weights,
	}
	if m is not None:
		data["text_masks"] = np.random.randn(k, m).astype(np.float32)

	np.savez(path, **data)


def run_self_test() -> None:
	"""Run complete self-test suite on synthetic data."""
	print("--- Running verify_rxr_stack Self-Test Suite ---")
	with tempfile.TemporaryDirectory() as temp_dir_str:
		temp_dir = Path(temp_dir_str)
		raw_split_dir = temp_dir / "pose_traces" / "rxr_train"
		raw_split_dir.mkdir(parents=True, exist_ok=True)

		combined_dir = temp_dir / "combined_output"

		print("1. Generating synthetic .npz pose_trace files...")
		generate_synthetic_npz(
			raw_split_dir / "000000_follower_pose_trace.npz", n=15, k=3, m=78
		)
		generate_synthetic_npz(
			raw_split_dir / "000000_guide_pose_trace.npz", n=20, k=4, m=85
		)
		generate_synthetic_npz(
			raw_split_dir / "000006_follower_pose_trace.npz", n=10, k=5, m=98
		)
		# Test file without text_masks
		generate_synthetic_npz(
			raw_split_dir / "000007_follower_pose_trace.npz", n=30, k=7, m=None
		)

		print("2. Running format_RxR_multinode stacker on synthetic data...")
		process_split(
			split_dir=raw_split_dir,
			split_name="rxr_train",
			destination_dir=combined_dir,
			node_index=0,
			node_count=1,
			sharding_mode="node",
		)

		print("3. Verifying generated HDF5 and JSON index against raw synthetic files...")
		h5_path = combined_dir / "rxr_train.h5"
		json_path = combined_dir / "rxr_train.json"

		traces_count, snapshots_count = verify_split_shard(
			h5_path=h5_path,
			json_path=json_path,
			raw_split_dir=raw_split_dir,
		)
		print(
			f"Self-Test Verification Complete: {traces_count} traces and {snapshots_count} snapshots verified exact match."
		)
		assert traces_count == 4, f"Expected 4 traces, verified {traces_count}"
		assert snapshots_count == (
			15 + 20 + 10 + 30
		), f"Expected 75 snapshots, verified {snapshots_count}"
		print("PASSED: Self-test succeeded with 100% numerical and structural fidelity.")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Verify RxR pose_traces HDF5 stack against raw .npz files."
	)
	parser.add_argument(
		"--combined-dataset",
		type=Path,
		help="Path to combined HDF5/JSON dataset output.",
	)
	parser.add_argument(
		"--extract-root",
		type=Path,
		help="Path to raw pose_traces root directory.",
	)
	parser.add_argument(
		"--run-self-test",
		action="store_true",
		help="Run standalone synthetic verification test suite.",
	)
	args = parser.parse_args()

	if args.run_self_test:
		run_self_test()
		return

	if not args.combined_dataset or not args.extract_root:
		print("Error: Must specify either --run-self-test OR both --combined-dataset and --extract-root.")
		sys.exit(1)

	pose_traces_root = locate_pose_traces_dir(args.extract_root)
	total_traces = 0
	total_snapshots = 0

	for json_path in sorted(args.combined_dataset.glob("*.json")):
		stem = json_path.stem
		# Handle node_X_split.json or split.json
		if stem.startswith("node_"):
			parts = stem.split("_")
			split_name = "_".join(parts[2:])
		else:
			split_name = stem

		h5_path = args.combined_dataset / f"{stem}.h5"
		raw_split_dir = pose_traces_root / split_name
		if not raw_split_dir.is_dir():
			print(f"Warning: Raw split directory not found: {raw_split_dir}")
			continue

		print(f"Verifying {stem}.h5 against raw files in {raw_split_dir}...")
		t_cnt, s_cnt = verify_split_shard(h5_path, json_path, raw_split_dir)
		total_traces += t_cnt
		total_snapshots += s_cnt

	print(f"\nOverall Verification Complete: {total_traces} traces, {total_snapshots} snapshots verified.")


if __name__ == "__main__":
	main()
