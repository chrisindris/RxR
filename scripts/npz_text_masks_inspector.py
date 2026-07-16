from pathlib import Path

import numpy as np

# as show in this file, we have these max dimensions:
# rxr_train: 136, 512
# rxr_val_seen: 95, 512
# rxr_val_unseen: 147, 512


def main():
	max_k = 0
	max_m = 0
	data_dir = Path("/scratch/indrisch/RxR_data/rxr-data/pose_traces/rxr_val_unseen/")

	for npz_path in data_dir.glob("*.npz"):
		try:
			with np.load(npz_path) as data:
				k, m = data["text_masks"].shape 
		except KeyError:
			print(f"Key 'text_masks' not found in {npz_path}")
			continue
		except EOFError:
			print(f"EOFError encountered while reading {npz_path}. The file may be corrupted.")
            # possibly corrupted files in rxr_train:
            # 118920_follower_pose_trace.npz
            # 071596_follower_pose_trace.npz
            # 071160_follower_pose_trace.npz
            # possibly corrupted files in rxr_val_seen:
            # 018895_follower_pose_trace.npz
            # possibly corrupted files in rxr_val_unseen:
            # 059987_follower_pose_trace.npz
            # 078826_follower_pose_trace.npz
			continue

		if k > max_k:
			max_k = k
		if m > max_m:
			max_m = m

	print(max_k)
	print(max_m)


if __name__ == "__main__":
	main()
