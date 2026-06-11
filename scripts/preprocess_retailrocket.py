#!/usr/bin/env python3
"""CLI chuyển đổi RetailRocket -> format huấn luyện cho từng model."""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ncs_paths import data_dir
from scripts.retailrocket_base import (
    save_core_inter,
    save_csgnn_pickle,
    save_duorec_inter,
    save_gce_graph,
    save_pickle_session,
)
from scripts.retailrocket_papers import (
    link_papers_datasets,
    save_cct_gnn,
    save_cm_hgnn,
    save_fgnn,
    save_hgcan,
)

NHOM_MODELS = [
    "SR-GNN",
    "GCE-GNN",
    "DHCN",
    "COTREC",
    "CSGNN",
    "DuoRec",
    "CORE",
    "SCL-DHCN",
    "SCL-COTREC",
    "SCL-GCE-GNN",
]

PAPER_MODELS = [
    "FGNN",
    "CM-HGNN",
    "CCT-GNN",
    "HGCAN",
]

PICKLE_MODELS = [
    "SR-GNN",
    "GCE-GNN",
    "DHCN",
    "COTREC",
    "CSGNN",
    "SCL-DHCN",
    "SCL-COTREC",
    "SCL-GCE-GNN",
]

MODELS = {
    "SR-GNN": lambda: save_pickle_session(data_dir("SR-GNN", "retailrocket")),
    "GCE-GNN": lambda: (
        save_pickle_session(data_dir("GCE-GNN", "retailrocket")),
        save_gce_graph(data_dir("GCE-GNN", "retailrocket"), 12),
    ),
    "DHCN": lambda: save_pickle_session(data_dir("DHCN", "retailrocket")),
    "COTREC": lambda: save_pickle_session(data_dir("COTREC", "retailrocket")),
    "CSGNN": lambda: save_csgnn_pickle(data_dir("CSGNN", "retailrocket")),
    "DuoRec": lambda: save_duorec_inter(data_dir("DuoRec", "retailrocket")),
    "CORE": lambda: save_core_inter(data_dir("CORE", "retailrocket")),
    "SCL-DHCN": lambda: save_pickle_session(
        data_dir("SelfContrastiveLearningRecSys", "retailrocket") / "DHCN"
    ),
    "SCL-COTREC": lambda: save_pickle_session(
        data_dir("SelfContrastiveLearningRecSys", "retailrocket") / "COTREC"
    ),
    "SCL-GCE-GNN": lambda: (
        save_pickle_session(data_dir("SelfContrastiveLearningRecSys", "retailrocket") / "GCE-GNN"),
        save_gce_graph(data_dir("SelfContrastiveLearningRecSys", "retailrocket") / "GCE-GNN", 12),
    ),
    "FGNN": lambda: save_fgnn(),
    "CM-HGNN": lambda: save_cm_hgnn(),
    "CCT-GNN": lambda: save_cct_gnn(),
    "HGCAN": lambda: save_hgcan(),
    "all-pickle": None,
    "all-papers": None,
    "all": None,
}


def _run(names):
    for name in names:
        print(f"\n=== {name} ===")
        MODELS[name]()


def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi Data/datagoc/Retailrocket/")
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=list(MODELS.keys()),
        help="Model cần chuyển đổi (all / all-papers / all-pickle / từng tên)",
    )
    parser.add_argument(
        "--link-papers",
        action="store_true",
        help="Tạo symlink papers_only/*/datasets/retailrocket -> Data/<Paper>/retailrocket",
    )
    args = parser.parse_args()

    if args.model == "all":
        _run(NHOM_MODELS + PAPER_MODELS)
        if args.link_papers:
            print("\n=== symlink papers_only ===")
            link_papers_datasets()
        print("\nHoàn tất tất cả model (nhom + papers_only).")
        return

    if args.model == "all-papers":
        _run(PAPER_MODELS)
        if args.link_papers:
            print("\n=== symlink papers_only ===")
            link_papers_datasets()
        return

    if args.model == "all-pickle":
        _run(PICKLE_MODELS)
        return

    MODELS[args.model]()
    if args.link_papers and args.model in PAPER_MODELS:
        link_papers_datasets()
    print("Xong.")


if __name__ == "__main__":
    main()
