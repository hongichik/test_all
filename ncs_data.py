"""Helper đọc data từ Data/<project>/<dataset>/ với fallback path cũ."""
import os
import pickle
from pathlib import Path

import numpy as np
from typing import List, Optional, Union

from ncs_paths import REPO_ROOT, data_dir, dataset_file


def _smoke_limit() -> int:
    if not os.environ.get("NCS_SMOKE"):
        return 0
    return int(os.environ.get("NCS_SMOKE_SAMPLES", "3000"))


def subsample_session_data(data):
    """Giới hạn số sample khi smoke test (NCS_SMOKE=1)."""
    n = _smoke_limit()
    if not n:
        return data
    if isinstance(data, tuple) and len(data) >= 2:
        return tuple(d[:n] for d in data)
    if isinstance(data, list):
        return data[:n]
    return data


def resolve_data_path(project: str, dataset: str, filename: str, legacy: Union[str, Path]) -> Path:
    """Ưu tiên Data/, fallback path legacy trong repo model."""
    canonical = dataset_file(project, dataset, filename)
    if canonical.exists():
        return canonical
    legacy_path = Path(legacy)
    if not legacy_path.is_absolute():
        legacy_path = REPO_ROOT / legacy_path
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(
        f"Không tìm thấy {filename} cho {project}/{dataset}.\n"
        f"  Đã thử: {canonical}\n"
        f"  Đã thử: {legacy_path}\n"
        f"  Chạy preprocess_retailrocket.py trước."
    )


def load_pickle(
    project: str,
    dataset: str,
    filename: str,
    legacy: Union[str, Path],
    *,
    subsample: bool = True,
):
    path = resolve_data_path(project, dataset, filename, legacy)
    with open(path, "rb") as f:
        data = pickle.load(f)
    if subsample:
        return subsample_session_data(data)
    return data


def count_nodes_from_pickle(train_data, *extra_datasets) -> int:
    """Suy ra số node từ max item id trong train/test (và all_train nếu có)."""
    max_id = 0

    def _update_from_seqs(seqs):
        nonlocal max_id
        for seq in seqs:
            if len(seq):
                max_id = max(max_id, int(max(seq)))

    def _update_from_targets(targets):
        nonlocal max_id
        for t in targets:
            max_id = max(max_id, int(t))

    def _update_from_dataset(data):
        if data is None:
            return
        if hasattr(data, "inputs"):
            _update_from_seqs(data.inputs)
            return
        if isinstance(data, tuple) and len(data) >= 1 and isinstance(data[0], (list, np.ndarray)):
            _update_from_seqs(data[0])
            if len(data) >= 2 and isinstance(data[1], (list, np.ndarray)):
                _update_from_targets(data[1])
            return
        if isinstance(data, list):
            if data and isinstance(data[0], (list, tuple, np.ndarray)):
                _update_from_seqs(data)
            else:
                for item in data:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        if item:
                            max_id = max(max_id, int(max(item)))
                    else:
                        max_id = max(max_id, int(item))

    _update_from_dataset(train_data)
    for extra in extra_datasets:
        _update_from_dataset(extra)
    return max_id + 1
