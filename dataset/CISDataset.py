import os
import random
import re
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

import mat73
import numpy as np
import torch
from scipy.io import loadmat, whosmat
from torch.utils.data import Dataset

_DATA_KEYS = ("sample_data", "Q_matrix", "Q_out", "S_out")
_AGG_FILENAME = "Q_all.mat"


def sort_key(name: str):
    tokens = re.split(r"(\d+)", str(name))
    return [int(tok) if tok.isdigit() else tok.lower() for tok in tokens]


def _load_single_mat(path: str) -> np.ndarray:
    try:
        data = mat73.loadmat(path, use_attrdict=True)
        for key in _DATA_KEYS:
            if key in data:
                return np.asarray(data[key])
        usable = [k for k in data.keys() if not k.startswith("__")]
        if usable:
            return np.asarray(data[usable[0]])
        raise KeyError(f"No data key found in {path}")
    except Exception:
        mat = loadmat(path, squeeze_me=False, struct_as_record=False)
        for key in _DATA_KEYS:
            if key in mat:
                return np.asarray(mat[key])
        usable = [k for k in mat.keys() if not k.startswith("__")]
        if not usable:
            raise KeyError(f"No usable key in {path}")
        return np.asarray(mat[usable[0]])


def _read_qall_metadata(path: str) -> Tuple[int, np.ndarray, np.ndarray]:
    q_shape = None
    for name, shape, _ in whosmat(path):
        if name == "Q_all":
            q_shape = shape
            break
    if q_shape is None:
        raise KeyError(f"'Q_all' not found in {path}")
    num_samples = int(q_shape[0])

    meta = loadmat(path, variable_names=["valid", "sampleIdx"], squeeze_me=False, struct_as_record=False)
    valid = meta.get("valid")
    if valid is None:
        valid_mask = np.ones(num_samples, dtype=bool)
    else:
        valid_mask = np.asarray(valid).reshape(-1).astype(bool)
        if valid_mask.size < num_samples:
            pad = np.ones(num_samples - valid_mask.size, dtype=bool)
            valid_mask = np.concatenate([valid_mask, pad])
        elif valid_mask.size > num_samples:
            valid_mask = valid_mask[:num_samples]

    sample_idx = meta.get("sampleIdx")
    if sample_idx is None:
        sample_idx = np.arange(num_samples, dtype=np.int32)
    else:
        sample_idx = np.asarray(sample_idx).reshape(-1)
        if sample_idx.size < num_samples:
            tail = np.arange(sample_idx.size, num_samples, dtype=np.int32)
            sample_idx = np.concatenate([sample_idx, tail])
        elif sample_idx.size > num_samples:
            sample_idx = sample_idx[:num_samples]

    return num_samples, valid_mask, sample_idx.astype(np.int32)


class _QAllCache:
    def __init__(self, max_entries: int = 4):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get_slice(self, path: str, row: int) -> np.ndarray:
        arr = self._cache.get(path)
        if arr is None:
            mat = loadmat(path, squeeze_me=False, struct_as_record=False)
            if "Q_all" not in mat:
                raise KeyError(f"'Q_all' not found in {path}")
            arr = np.asarray(mat["Q_all"])
            self._cache[path] = arr
            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(path)

        if row >= arr.shape[0]:
            raise IndexError(f"Row {row} out of range for {path}")
        return arr[row]


def _gather_samples(data_root: str) -> List[Dict]:
    samples: List[Dict] = []
    for dist_name in sorted(os.listdir(data_root), key=sort_key):
        dist_path = os.path.join(data_root, dist_name)
        if not os.path.isdir(dist_path):
            continue

        card_folders = sorted(
            [c for c in os.listdir(dist_path) if os.path.isdir(os.path.join(dist_path, c))],
            key=sort_key
        )

        for card_idx, card_name in enumerate(card_folders):
            card_path = os.path.join(dist_path, card_name)
            qall_path = os.path.join(card_path, _AGG_FILENAME)

            if os.path.isfile(qall_path):
                try:
                    total, valid_mask, sample_idx = _read_qall_metadata(qall_path)
                except Exception as exc:
                    raise RuntimeError(f"Failed to read metadata from {qall_path}") from exc

                for row in range(total):
                    if not valid_mask[row]:
                        continue
                    samples.append({
                        "kind": "aggregate",
                        "path": qall_path,
                        "row": row,
                        "label_idx": card_idx,
                        "label_name": card_name,
                        "distance": dist_name,
                        "sample_id": int(sample_idx[row])
                    })
                continue

            for file in sorted(os.listdir(card_path), key=sort_key):
                if not file.lower().endswith(".mat"):
                    continue
                file_path = os.path.join(card_path, file)
                samples.append({
                    "kind": "file",
                    "path": file_path,
                    "row": None,
                    "label_idx": card_idx,
                    "label_name": card_name,
                    "distance": dist_name,
                    "sample_id": None
                })

    if not samples:
        raise RuntimeError(f"No .mat samples found under {data_root}")
    return samples


class CISDataset(Dataset):
    def __init__(self, data_root: str, cache_size: int = 4):
        self.data_root = data_root
        self.samples = _gather_samples(data_root)
        self._cache = _QAllCache(max_entries=cache_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        if sample["kind"] == "aggregate":
            data = self._cache.get_slice(sample["path"], sample["row"])
        else:
            data = _load_single_mat(sample["path"])

        cis = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return cis, sample["label_idx"], sample["distance"]


class TripletCISDataset(Dataset):
    def __init__(self, data_root: str):
        self.base = CISDataset(data_root)
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, sample in enumerate(self.base.samples):
            self.label_to_indices[sample["label_idx"]].append(idx)
        if any(len(idxs) < 2 for idxs in self.label_to_indices.values()):
            raise ValueError("Each class must contain at least two samples for triplet mining.")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        anchor_cis, anchor_label, _ = self.base[idx]
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.choice(self.label_to_indices[anchor_label])
        negative_label = random.choice([l for l in self.label_to_indices.keys() if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])

        positive_cis = self.base[positive_idx][0]
        negative_cis = self.base[negative_idx][0]
        return anchor_cis, positive_cis, negative_cis


class RxAgnosticCISDataset(Dataset):
    def __init__(self, data_root: str, cache_size: int = 4):
        self.data_root = data_root
        self.samples = _gather_samples(data_root)
        self._cache = _QAllCache(max_entries=cache_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        if sample["kind"] == "aggregate":
            data = self._cache.get_slice(sample["path"], sample["row"])
        else:
            data = _load_single_mat(sample["path"])

        cis = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return cis, sample["label_name"], sample["distance"]
