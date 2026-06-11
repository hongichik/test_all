"""Đường dẫn chuẩn NCS — data tập trung tại Data/."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "Data"
RAW_RETAILROCKET = DATA_ROOT / "datagoc" / "Retailrocket"

# Ánh xạ tên bài toán -> thư mục code
PROJECT_DIRS = {
    "SR-GNN": REPO_ROOT / "nhom1" / "SR-GNN",
    "GCE-GNN": REPO_ROOT / "nhom1" / "GCE-GNN",
    "DHCN": REPO_ROOT / "nhom2" / "DHCN",
    "COTREC": REPO_ROOT / "nhom2" / "COTREC",
    "CSGNN": REPO_ROOT / "nhom2" / "CSGNN",
    "DuoRec": REPO_ROOT / "nhom3" / "DuoRec",
    "SelfContrastiveLearningRecSys": REPO_ROOT / "nhom3" / "SelfContrastiveLearningRecSys",
    "CORE": REPO_ROOT / "nhom3" / "CORE",
    "NCL": REPO_ROOT / "nhom3" / "NCL",
}


def data_dir(project: str, dataset: str = "retailrocket") -> Path:
    """Data huấn luyện: Data/<project>/<dataset>/"""
    return DATA_ROOT / project / dataset


def project_code_dir(project: str) -> Path:
    return PROJECT_DIRS[project]


def dataset_file(project: str, dataset: str, filename: str) -> Path:
    return data_dir(project, dataset) / filename
