"""Helper đọc data từ Data/<project>/<dataset>/ với fallback path cũ."""
import pickle
from pathlib import Path
from typing import List, Optional, Union

from ncs_paths import REPO_ROOT, data_dir, dataset_file


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


def load_pickle(project: str, dataset: str, filename: str, legacy: Union[str, Path]):
    path = resolve_data_path(project, dataset, filename, legacy)
    with open(path, "rb") as f:
        return pickle.load(f)


def count_nodes_from_pickle(train_data, all_train=None) -> int:
    """Suy ra số node từ max item id trong train/test."""
    max_id = 0
    for seqs, _ in [train_data]:
        for seq in seqs:
            if seq:
                max_id = max(max_id, max(seq))
    if all_train:
        for seq in all_train:
            if seq:
                max_id = max(max_id, max(seq))
    return max_id + 1
