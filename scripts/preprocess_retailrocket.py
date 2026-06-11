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
    "all-pickle": None,
    "all": None,
}


def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi Data/datagoc/Retailrocket/")
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=list(MODELS.keys()),
        help="Model cần chuyển đổi (hoặc all / all-pickle)",
    )
    args = parser.parse_args()

    if args.model == "all":
        for name in ["SR-GNN", "GCE-GNN", "DHCN", "COTREC", "CSGNN", "DuoRec", "CORE",
                     "SCL-DHCN", "SCL-COTREC", "SCL-GCE-GNN"]:
            print(f"\n=== {name} ===")
            MODELS[name]()
        print("\nHoàn tất tất cả model.")
        return

    if args.model == "all-pickle":
        for name in ["SR-GNN", "GCE-GNN", "DHCN", "COTREC", "CSGNN",
                     "SCL-DHCN", "SCL-COTREC", "SCL-GCE-GNN"]:
            print(f"\n=== {name} ===")
            MODELS[name]()
        return

    MODELS[args.model]()
    print("Xong.")


if __name__ == "__main__":
    main()
