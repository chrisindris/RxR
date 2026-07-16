Each .npz contains matrices of arbitrary shape.

We should be able to stack these in the following way (and with a json); note that we will have separate stacks and jsons for each of "rxr_train", "rxr_val_seen", "rxr_val_unseen":

* json will have a key for each of the present 6-digit scene keys (i.e. 000000)
* in each of the above, two main keys, "follower" and "guide" (and also two keys k_max and m_max for indexing text_masks). 
* in each of the above, we will store the (n_index, k_index, m_index) (which is basically the cumulative of all the ones seen thus far up to that point), as well as (n, k, m) so that we know where they terminate. Between these two, we will be able to extract them. Note that not every file has "text_masks".

For example, each schema is as follows:
```python
{'pano': (np.str, [n, 1]), # each entry, listed as "np.str", appears to be a 32-digit hex number
 'time': (np.float32, [n, 1]),
 'audio_time': (np.float32, [n, 1]),
 'extrinsic_matrix': (np.float32, [n, 16]),
 'intrinsic_matrix': (np.float32, [n, 16]),
 'image_mask': (np.bool, [k, 128, 256]),
 'text_masks': (np.bool, [k, m]),
 'feature_weights': (np.float32, [k, 36])}
```
Where `n` is the number of snapshots, `k` is the number of panoramic viewpoints
in the associated path, and `m` is the number of BERT SubWord in the tokenized
instructions...

We can therefore stack the tensors into the following where X represents the total number of .npz files we are stacking into the stack, (N, K, M) are the sum total of all (n, k, m) across all files respectively; note that k_max and m_max for the different jsons "rxr_train", "rxr_val_seen", "rxr_val_unseen" are known from npz_text_masks_inspector to be (rxr_train: 136, 512; rxr_val_seen: 95, 512; rxr_val_unseen: 147, 512):

```python
{'pano': (np.str, [N, 1]),
 'time': (np.float32, [N, 1]),
 'audio_time': (np.float32, [N, 1]),
 'extrinsic_matrix': (np.float32, [N, 16]),
 'intrinsic_matrix': (np.float32, [N, 16]),
 'image_mask': (np.bool, [K, 128, 256]),
 'text_masks': (np.bool, [k_max, m_max, X]),
 'feature_weights': (np.float32, [K, 36])}
```


Here are some examples:
```bash
(venv_RxR) [indrisch@tri-login01 scripts]$ python npz_inspector.py /scratch/indrisch/RxR_data/rxr-data/pose_traces/rxr_train/000000_follower_pose_trace.npz
Keys: ['pano', 'time', 'audio_time', 'extrinsic_matrix', 'intrinsic_matrix', 'image_mask', 'feature_weights', 'text_masks']
Key: pano | Shape: (2334,) | Dtype: <U32
Key: time | Shape: (2334,) | Dtype: float64
Key: audio_time | Shape: (2334,) | Dtype: float64
Key: extrinsic_matrix | Shape: (2334, 4, 4) | Dtype: float64
Key: intrinsic_matrix | Shape: (2334, 4, 4) | Dtype: float64
Key: image_mask | Shape: (3, 128, 256) | Dtype: bool
Key: feature_weights | Shape: (3, 36) | Dtype: float32
Key: text_masks | Shape: (3, 78) | Dtype: float32
(venv_RxR) [indrisch@tri-login01 scripts]$ 
(venv_RxR) [indrisch@tri-login01 scripts]$ python npz_inspector.py /scratch/indrisch/RxR_data/rxr-data/pose_traces/rxr_train/000006_follower_pose_trace.npz
Keys: ['pano', 'time', 'audio_time', 'extrinsic_matrix', 'intrinsic_matrix', 'image_mask', 'feature_weights', 'text_masks']
Key: pano | Shape: (6154,) | Dtype: <U32
Key: time | Shape: (6154,) | Dtype: float64
Key: audio_time | Shape: (6154,) | Dtype: float64
Key: extrinsic_matrix | Shape: (6154, 4, 4) | Dtype: float64
Key: intrinsic_matrix | Shape: (6154, 4, 4) | Dtype: float64
Key: image_mask | Shape: (5, 128, 256) | Dtype: bool
Key: feature_weights | Shape: (5, 36) | Dtype: float32
Key: text_masks | Shape: (5, 98) | Dtype: float32
(venv_RxR) [indrisch@tri-login01 scripts]$ python npz_inspector.py /scratch/indrisch/RxR_data/rxr-data/pose_traces/rxr_train/000007_follower_pose_trace.npz
Keys: ['pano', 'time', 'audio_time', 'extrinsic_matrix', 'intrinsic_matrix', 'image_mask', 'feature_weights', 'text_masks']
Key: pano | Shape: (3333,) | Dtype: <U32
Key: time | Shape: (3333,) | Dtype: float64
Key: audio_time | Shape: (3333,) | Dtype: float64
Key: extrinsic_matrix | Shape: (3333, 4, 4) | Dtype: float64
Key: intrinsic_matrix | Shape: (3333, 4, 4) | Dtype: float64
Key: image_mask | Shape: (7, 128, 256) | Dtype: bool
Key: feature_weights | Shape: (7, 36) | Dtype: float32
Key: text_masks | Shape: (7, 99) | Dtype: float32
```