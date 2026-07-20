# RxR Combined H5 Dataset — PyTorch Dataloader Spec

This document is written for a coding LLM (or engineer) implementing a
**PyTorch `Dataset` / `DataLoader`** over the compressed RxR pose-trace dataset
at:

```text
/scratch/indrisch/RxR_data_combined_h5_multinode/
```

The intended downstream use is deep-learning training via
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). This file specifies
**how to locate and reconstruct individual pose traces** from the on-disk H5/JSON
layout. It does **not** prescribe a particular chat/SFT JSON schema; map the
returned tensors into whatever LLaMA-Factory sample format your training stage
needs after loading.

Related sources of truth in this repo:

| Path | Role |
|------|------|
| `docs/pose_traces.md` | Original stacking design (per-trace schema, cumulative indices) |
| `scripts/format_RxR_multinode.py` | How `.npz` pose traces were packed into H5 + per-node JSON |
| `scripts/format_RxR_multinode_wrapper.sh` | Multinode SLURM entry for packing |
| `scripts/create_entryjson_RxR_multinode.py` | Builds the global `rxr.json` entry point |
| `scripts/verify_rxr_stack.py` | Reference reconstruction / equality checks against raw `.npz` |

---

## 1. Why this layout exists

The raw RxR pose traces are many small `.npz` files under
`pose_traces/{rxr_train,rxr_val_seen,rxr_val_unseen}/`. High file counts are
painful on shared filesystems. The packing pipeline concatenates many traces into
a small number of **gzip-compressed HDF5 shards**, with **JSON index files**
holding cumulative slice bounds so any original trace can be recovered exactly.

Pipeline:

1. `format_RxR_multinode_wrapper.sh` → `format_RxR_multinode.py`  
   Produces `node_{N}_rxr_{split}.h5` + `node_{N}_rxr_{split}.json` for each
   SLURM node and split.
2. `create_entryjson_RxR_multinode_wrapper.sh` → `create_entryjson_RxR_multinode.py`  
   Produces the global entry map `rxr.json`.

---

## 2. On-disk layout

**Root (canonical):** `/scratch/indrisch/RxR_data_combined_h5_multinode/`

```text
RxR_data_combined_h5_multinode/
├── rxr.json                          # global entry map (start here)
├── node_0_rxr_train.h5
├── node_0_rxr_train.json
├── node_0_rxr_val_seen.h5
├── node_0_rxr_val_seen.json
├── node_0_rxr_val_unseen.h5
├── node_0_rxr_val_unseen.json
├── node_1_rxr_train.h5
├── ...
└── node_7_rxr_val_unseen.json
```

### 2.1 Naming conventions

| Concept | Value |
|---------|--------|
| Node index `N` | Integer `0` … `7` (8 shards in the current build) |
| Split in **filenames** | `train`, `val_seen`, `val_unseen` after the `rxr_` prefix |
| Shard basename pattern | `node_{N}_rxr_{split}.h5` / `.json` |
| Split keys in **`rxr.json`** | `"train"`, `"val_seen"`, `"val_unseen"` (no `rxr_` prefix) |

Example paths for node 0, train:

```text
node_0_rxr_train.h5
node_0_rxr_train.json
```

### 2.2 Approximate contents of the current build

| Split | Follower traces | Guide traces | Total traces | Unique 6-digit IDs |
|-------|----------------:|-------------:|-------------:|-------------------:|
| `train` | ~79 464 | ~79 467 | ~158 931 | ~79 467 |
| `val_seen` | ~8 812 | ~8 813 | ~17 625 | ~8 813 |
| `val_unseen` | ~13 650 | ~13 652 | ~27 302 | ~13 652 |

Most 6-digit IDs appear **twice** across the dataset: once as a **follower**
trace and once as a **guide** trace (often on **different** node shards because
files were round-robin sharded after sorting). See §5 for implications for
`rxr.json`.

---

## 3. Global entry file: `rxr.json`

**Path:** `{dataset_root}/rxr.json`

**Purpose (from `create_entryjson_RxR_multinode.py`):** single file that maps a
split + 6-digit scene key to the **node index** whose per-node JSON/H5 holds that
key. Example:

```python
# Conceptual lookup intended by the entry JSON
node = rxr_json["train"]["000000"]   # e.g. 0
# then open:
#   node_0_rxr_train.json
#   node_0_rxr_train.h5
```

### 3.1 Schema

```json
{
  "train": {
    "000000": 0,
    "000006": 2,
    "000007": 4
  },
  "val_seen": {
    "000026": 1
  },
  "val_unseen": {
    "000048": 6
  }
}
```

- Top-level keys: `"train" | "val_seen" | "val_unseen"`.
- Second-level keys: zero-padded 6-digit strings (`"000000"`, …).
- Values: integer node index `N` → use files
  `node_{N}_rxr_{split}.json` and `node_{N}_rxr_{split}.h5`.

### 3.2 Filename construction helper

```python
def shard_paths(dataset_root: str, split: str, node: int) -> tuple[str, str]:
    """split is one of: train | val_seen | val_unseen"""
    base = f"node_{node}_rxr_{split}"
    return (
        f"{dataset_root}/{base}.json",
        f"{dataset_root}/{base}.h5",
    )
```

### 3.3 Known caveats of the current `rxr.json` (implementers must read)

1. **Guide vs follower share the same 6-digit key space.**  
   Guide files are named `{instruction_id:06}_guide_pose_trace.npz` and follower
   files `{demonstration_id:06}_follower_pose_trace.npz`. When the numeric IDs
   coincide (common), the **same** key (e.g. `"000000"`) appears in **two** node
   JSONs: one with role `"follower"`, one with role `"guide"`.

2. **`create_entryjson` last-write-wins per key.**  
   The amalgamation loop is effectively:

   ```python
   for each node_*_rxr_{split}.json:
       for key in data:
           D[split][key] = node_num
   ```

   So `rxr.json[split][id]` points at **only one** of the two shards. Looking up
   only that node **misses the other role** for that ID.

3. **`k_max` / `m_max` leak into `rxr.json`.**  
   Per-node JSONs store `"k_max"` and `"m_max"` as top-level keys. The entry
   builder does not skip them, so `rxr.json[split]` may contain:

   ```python
   "k_max": <some node int>,  # NOT the true k_max
   "m_max": <some node int>,  # NOT the true m_max
   ```

   Always skip non-scene keys when iterating samples:
   `if key in ("k_max", "m_max") or not key.isdigit(): continue`.

**Practical recommendation for a complete training Dataset:**  
enumerate samples by scanning **all** `node_*_rxr_{split}.json` files (see §6).
Use `rxr.json` only as an optional hint when you already know
`(split, scene_id, role)` and have verified that role lives on the mapped node;
or rebuild a richer entry map keyed by `(scene_id, role)`.

---

## 4. Per-node JSON index

**Path:** `node_{N}_rxr_{split}.json`

### 4.1 Top-level structure

```json
{
  "k_max": 136,
  "m_max": 512,
  "000000": {
    "follower": { "...meta..." }
  },
  "000009": {
    "follower": { "...meta..." }
  }
}
```

| Key | Meaning |
|-----|---------|
| `k_max` | Max panoramic viewpoints used to pad `text_masks` in this split |
| `m_max` | Max BERT subword length used to pad `text_masks` in this split |
| `"######"` | Scene / annotation id (6-digit string) |

### 4.2 Split-wise padding constants (authoritative)

From `format_RxR_multinode.SPLIT_MAX_DIMS` (also stored as `k_max` / `m_max` in
each node JSON for that split):

| Split | `k_max` | `m_max` |
|-------|--------:|--------:|
| `train` | 136 | 512 |
| `val_seen` | 95 | 512 |
| `val_unseen` | 147 | 512 |

### 4.3 Per-trace metadata object

Under each scene id, keys are roles: `"follower"` and/or `"guide"` (in practice
a given node JSON almost always has **exactly one** role per scene id; the other
role for the same id lives on another node).

```json
{
  "file_name": "000000_follower_pose_trace.npz",
  "n_index": 0,
  "n": 2334,
  "k_index": 0,
  "k": 3,
  "x_index": 0,
  "m": 78,
  "has_text_masks": true,
  "has_audio_time": true
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `file_name` | str | Original `.npz` basename (for debugging / verification only) |
| `n_index` | int | Start index along the **snapshot** axis (`N`) in H5 |
| `n` | int | Number of snapshots in this trace |
| `k_index` | int | Start index along the **panorama** axis (`K`) in H5 |
| `k` | int | Number of panoramic viewpoints in this trace |
| `x_index` | int | Index along the **trace** axis of `text_masks` (`X`); **`-1` if no text masks** |
| `m` | int | True text-mask width (subwords) before padding; `0` if absent |
| `has_text_masks` | bool | Whether `text_masks` existed in the source `.npz` |
| `has_audio_time` | bool | Whether source `.npz` had `audio_time` (followers: typically true; guides: false) |

**Slice end indices (exclusive):**

```text
snapshots:   [n_index, n_index + n)
panoramas:   [k_index, k_index + k)
text mask:   row x_index  (if has_text_masks and x_index >= 0)
             valid columns [0, m), valid rows [0, k) inside the padded (k_max, m_max) plane
```

---

## 5. Per-node HDF5 layout

**Path:** `node_{N}_rxr_{split}.h5`  
**Library:** `h5py` (read-only is sufficient)

### 5.1 Datasets

Created by `format_RxR_multinode.process_split` (all use gzip compression level 9):

| Dataset | Stored shape | dtype | Axis meaning |
|---------|--------------|-------|--------------|
| `pano` | `(N_total, 1)` | UTF-8 string (`h5py` string dtype) | Snapshot |
| `time` | `(N_total, 1)` | `float64` | Snapshot |
| `audio_time` | `(N_total, 1)` | `float64` | Snapshot (zeros if source lacked audio) |
| `extrinsic_matrix` | `(N_total, 16)` | `float64` | Snapshot; row-major flatten of `(4, 4)` |
| `intrinsic_matrix` | `(N_total, 16)` | `float64` | Snapshot; row-major flatten of `(4, 4)` |
| `image_mask` | `(K_total, 128, 256)` | `bool` | Panorama |
| `feature_weights` | `(K_total, 36)` | `float32` | Panorama |
| `text_masks` | `(X_total, k_max, m_max)` | `float32` | One padded plane **per trace that had text_masks** |

`N_total`, `K_total`, `X_total` differ per shard.

### 5.2 Semantic fields (from RxR README / `pose_traces.md`)

| Field | Description |
|-------|-------------|
| `pano` | Panoramic viewpoint id for each snapshot (32-char hex string) |
| `time` | Snapshot timestamp (seconds) |
| `audio_time` | Follower progress through the guide audio; **guide traces omit this in the original data** (stored as zeros in H5 when missing) |
| `extrinsic_matrix` | Camera pose, \(4\times4\) |
| `intrinsic_matrix` | Camera intrinsics / projection, \(4\times4\) |
| `image_mask` | Equirectangular FOV mask `[k, 128, 256]` |
| `text_masks` | Per-panorama mask over instruction subwords `[k, m]` (padded to `[k_max, m_max]` in H5) |
| `feature_weights` | Mean-pooled FOV weights over 36 discrete views `[k, 36]` |

Original per-file schema (before stacking):

```python
{
  "pano":              (str,     [n]),          # or [n, 1] after pack
  "time":              (float,   [n]),
  "audio_time":        (float,   [n]),          # follower only in source
  "extrinsic_matrix":  (float,   [n, 4, 4]),    # stored flat [n, 16] in H5
  "intrinsic_matrix":  (float,   [n, 4, 4]),    # stored flat [n, 16] in H5
  "image_mask":        (bool,    [k, 128, 256]),
  "text_masks":        (float,   [k, m]),       # optional; often float32 in source
  "feature_weights":   (float32, [k, 36]),
}
```

---

## 6. Reconstruction algorithm (required for any correct loader)

This matches `verify_rxr_stack.verify_split_shard`.

### 6.1 Inputs

- `dataset_root`, `split` (`train` | `val_seen` | `val_unseen`)
- `node` (int)
- `scene_id` (6-digit str)
- `role` (`"follower"` | `"guide"`)

### 6.2 Steps

```python
import json
from pathlib import Path

import h5py
import numpy as np


def load_pose_trace(
    dataset_root: str | Path,
    split: str,
    node: int,
    scene_id: str,
    role: str,
) -> dict[str, np.ndarray | None]:
    dataset_root = Path(dataset_root)
    json_path = dataset_root / f"node_{node}_rxr_{split}.json"
    h5_path = dataset_root / f"node_{node}_rxr_{split}.h5"

    with json_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    k_max = int(index["k_max"])
    m_max = int(index["m_max"])
    meta = index[scene_id][role]

    n_index = meta["n_index"]
    n = meta["n"]
    k_index = meta["k_index"]
    k = meta["k"]
    x_index = meta["x_index"]
    m = meta["m"]
    has_text_masks = meta["has_text_masks"]
    has_audio_time = meta.get("has_audio_time", True)

    with h5py.File(h5_path, "r") as h5:
        pano = h5["pano"][n_index : n_index + n].reshape(-1)
        # Decode bytes → str if needed
        if pano.dtype.kind in ("S", "O"):
            pano = np.array(
                [
                    x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                    for x in pano
                ],
                dtype=object,
            )

        time = h5["time"][n_index : n_index + n].reshape(-1)
        audio_time = h5["audio_time"][n_index : n_index + n].reshape(-1)

        extrinsic = h5["extrinsic_matrix"][n_index : n_index + n].reshape(n, 4, 4)
        intrinsic = h5["intrinsic_matrix"][n_index : n_index + n].reshape(n, 4, 4)

        image_mask = h5["image_mask"][k_index : k_index + k]          # (k, 128, 256)
        feature_weights = h5["feature_weights"][k_index : k_index + k]  # (k, 36)

        text_masks = None
        if has_text_masks and x_index >= 0:
            # H5 stores zeros outside the true [k, m] region
            pad_k = min(k, k_max)
            pad_m = min(m, m_max)
            text_masks = h5["text_masks"][x_index, :pad_k, :pad_m].copy()

    if not has_audio_time:
        # Source had no audio_time; H5 may hold zeros — caller may set to None
        pass

    return {
        "scene_id": scene_id,
        "role": role,
        "split": split,
        "node": node,
        "file_name": meta["file_name"],
        "pano": pano,
        "time": time,
        "audio_time": audio_time if has_audio_time else None,
        "extrinsic_matrix": extrinsic,
        "intrinsic_matrix": intrinsic,
        "image_mask": image_mask,
        "feature_weights": feature_weights,
        "text_masks": text_masks,
        "n": n,
        "k": k,
        "m": m,
    }
```

### 6.3 Building the full sample index (recommended)

Do **not** rely only on `rxr.json` if you need every follower **and** guide trace.

```python
def build_sample_index(dataset_root: str | Path, split: str) -> list[dict]:
    """Return one record per pose trace for the split."""
    dataset_root = Path(dataset_root)
    samples: list[dict] = []
    for json_path in sorted(dataset_root.glob(f"node_*_rxr_{split}.json")):
        # node_3_rxr_train.json → 3
        node = int(json_path.name.split("_")[1])
        with json_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        for scene_id, roles in index.items():
            if scene_id in ("k_max", "m_max"):
                continue
            if not isinstance(roles, dict):
                continue
            for role, meta in roles.items():
                if role not in ("follower", "guide"):
                    continue
                samples.append(
                    {
                        "split": split,
                        "node": node,
                        "scene_id": scene_id,
                        "role": role,
                        # Optional: copy meta fields used in __getitem__ without re-opening JSON
                        **meta,
                    }
                )
    return samples
```

Optional filter examples:

```python
samples = [s for s in samples if s["role"] == "follower"]  # follower-only
samples = [s for s in samples if s["has_text_masks"]]      # skip rare missing masks
```

---

## 7. Suggested PyTorch `Dataset` design

### 7.1 Class sketch

```python
from torch.utils.data import Dataset


class RxRPoseTraceH5Dataset(Dataset):
    """
    One item = one pose trace (follower or guide) reconstructed from H5 slices.
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        roles: tuple[str, ...] = ("follower", "guide"),
        keep_file_handles: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.samples = [
            s for s in build_sample_index(self.dataset_root, split)
            if s["role"] in roles
        ]
        self.keep_file_handles = keep_file_handles
        self._h5: dict[int, h5py.File] = {}  # node → open handle (per-process)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_h5(self, node: int) -> h5py.File:
        if node not in self._h5:
            path = self.dataset_root / f"node_{node}_rxr_{self.split}.h5"
            self._h5[node] = h5py.File(path, "r")
        return self._h5[node]

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        # Prefer slicing via cached handle; same math as load_pose_trace()
        ...

    def __getstate__(self):
        # Critical for DataLoader workers: do not pickle open H5 handles
        state = self.__dict__.copy()
        state["_h5"] = {}
        return state

    def close(self) -> None:
        for f in self._h5.values():
            f.close()
        self._h5.clear()
```

### 7.2 Multiprocessing / H5 pitfalls (must follow)

1. **Never share one `h5py.File` across DataLoader worker processes.**  
   Open files lazily **inside each worker** (e.g. on first `__getitem__` after
   fork), or open/close per access (slower).

2. **Do not pickle open handles.** Implement `__getstate__` that clears `_h5`,
   or open files only in worker-local storage keyed by `id(self)` /
   `torch.utils.data.get_worker_info()`.

3. **Compression.** Datasets are gzip-compressed. Random access works but is
   heavier than raw `.npz`. Prefer:
   - moderate `num_workers` (e.g. 2–8),
   - `persistent_workers=True` when `num_workers > 0`,
   - caching open handles per worker,
   - optional sample order that improves locality within a shard.

4. **Variable-length batching.** `n` and `k` vary widely (train `n` can be tens
   of thousands; `k` up to `k_max`). You almost certainly need a custom
   `collate_fn` that pads/packs sequences, or use `batch_size=1` and accumulate
   gradients, or subsample snapshots/panoramas inside `__getitem__`.

5. **Memory.** Loading a full high-`n` trace returns large arrays (e.g. `n≈10^4`
   extrinsic matrices). For LLaMA-Factory-style training you may only need a
   **subset** of fields (e.g. panorama sequence + masks, not every snapshot).

### 7.3 Example `collate_fn` policy (choose one; document your choice in code)

| Policy | When to use |
|--------|-------------|
| `batch_size=1`, return dict of tensors as-is | Simplest; good for long traces |
| Pad snapshots to `max_n` in batch; pad panoramas to `max_k` | Small batches of short traces |
| Subsample / window `n` and/or `k` inside Dataset | Fixed-size model inputs |
| Return only metadata + paths, load lazily in collate | Unusual; usually worse with H5 |

---

## 8. LLaMA-Factory integration notes

LLaMA-Factory’s default data path is JSON/JSONL described in
`data/dataset_info.json` (alpaca / sharegpt / etc.). This H5 corpus is **not**
that format.

Recommended integration patterns:

1. **Custom PyTorch Dataset plugged into a custom training loop / LF extension**  
   Use `RxRPoseTraceH5Dataset` as the ground-truth loader; convert each item into
   the tensors/messages your model expects inside `__getitem__` or a collator.

2. **Offline export to JSONL** (only if samples are small text/feature records)  
   Iterate the H5 dataset once, write LLaMA-Factory-compatible JSONL, register in
   `dataset_info.json`. **Do not** dump full high-frequency pose streams into
   JSONL — keep heavy arrays in H5 and only export IDs + text side-channels if
   you add those later.

3. **Hybrid**  
   JSONL lists `(split, node, scene_id, role)` (or just `(split, scene_id, role)`
   with a fixed node map); the Dataset resolves tensors from H5 at train time.

Whatever path you choose, **the H5 slice math in §6 is the source of truth** for
pose-trace tensors.

Dependencies typically required:

```text
torch
numpy
h5py
```

On Compute Canada-style clusters used by this project, these are often installed
via the project virtualenv pattern in `scripts/format_RxR_multinode.sh`
(`numpy`, `torch`, `h5py`, … with `--no-index`).

---

## 9. Optional: lookup via `rxr.json` (partial)

When you only need a single known id and accept the last-write-wins limitation:

```python
def resolve_node(dataset_root: Path, split: str, scene_id: str) -> int:
    with (dataset_root / "rxr.json").open() as f:
        rxr = json.load(f)
    if scene_id in ("k_max", "m_max"):
        raise KeyError(scene_id)
    return int(rxr[split][scene_id])


def try_load_role(dataset_root, split, scene_id, role):
    """Try the rxr.json node first; if role missing, scan all nodes for that id."""
    node = resolve_node(dataset_root, split, scene_id)
    json_path = Path(dataset_root) / f"node_{node}_rxr_{split}.json"
    with json_path.open() as f:
        index = json.load(f)
    if role in index.get(scene_id, {}):
        return load_pose_trace(dataset_root, split, node, scene_id, role)
    # Fallback: search other shards (needed for the other role of the same id)
    for json_path in Path(dataset_root).glob(f"node_*_rxr_{split}.json"):
        node_i = int(json_path.name.split("_")[1])
        with json_path.open() as f:
            index = json.load(f)
        if role in index.get(scene_id, {}):
            return load_pose_trace(dataset_root, split, node_i, scene_id, role)
    raise KeyError(f"{split}/{scene_id}/{role} not found")
```

For training, prefer `build_sample_index` (§6.3) over repeated global scans.

---

## 10. Acceptance checks for a correct implementation

An implementation is correct if:

1. For a random sample of traces, arrays match the original `.npz` (same checks as
   `scripts/verify_rxr_stack.py`):
   - `pano`, `image_mask` exact equality  
   - `time`, `audio_time` (when present), matrices, `feature_weights`, `text_masks`
     with `assert_allclose`
2. `extrinsic_matrix` / `intrinsic_matrix` are reshaped to `(n, 4, 4)`.
3. `text_masks` is `None` (or omitted) when `has_text_masks` is false / `x_index == -1`.
4. Guide traces do not claim real `audio_time` (`has_audio_time` is false).
5. Dataset length equals the number of role entries across all node JSONs for the
   chosen split/role filter (see §2.2 for ballpark counts).
6. Multi-worker `DataLoader` iteration completes without HDF5 file-handle /
   pickling errors.

Minimal smoke test:

```python
ds = RxRPoseTraceH5Dataset(
    "/scratch/indrisch/RxR_data_combined_h5_multinode",
    split="train",
    roles=("follower",),
)
item = ds[0]
assert item["extrinsic_matrix"].shape == (item["n"], 4, 4)
assert item["image_mask"].shape == (item["k"], 128, 256)
assert item["feature_weights"].shape == (item["k"], 36)
if item["text_masks"] is not None:
    assert item["text_masks"].shape[0] == item["k"]
    assert item["text_masks"].shape[1] == item["m"]
```

---

## 11. Quick reference card

```text
Entry (optional / partial):  {root}/rxr.json
                             rxr[split][scene_id] -> node N

Shard index:                 {root}/node_{N}_rxr_{split}.json
Shard tensors:               {root}/node_{N}_rxr_{split}.h5

Sample identity:             (split, node, scene_id, role)

Snapshots axis N:            pano, time, audio_time, extrinsic_matrix, intrinsic_matrix
                             slice [n_index : n_index + n]

Panorama axis K:             image_mask, feature_weights
                             slice [k_index : k_index + k]

Text-mask axis X:            text_masks[x_index, :k, :m]   if has_text_masks
                             else skip (x_index == -1)

Reshape:                     extrinsic/intrinsic (n, 16) -> (n, 4, 4)
```

---

## 12. Out of scope (not in this H5 pack)

The combined H5 dataset currently packs **pose traces only**. It does **not**
include:

- guide / follower JSONL annotations (`instruction`, `path`, `language`, …)
- Matterport3D RGB panoramas
- precomputed BERT `text_features`

If training needs language instructions or images, load those from the original
RxR release (or other prepared assets) and **join** on `instruction_id` /
`demonstration_id` (the 6-digit `scene_id` + `role` in this pack).
