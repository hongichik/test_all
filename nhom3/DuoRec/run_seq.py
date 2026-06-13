import argparse
import os
import sys
import tempfile
from pathlib import Path

_duorec_root = os.path.dirname(os.path.abspath(__file__))
if _duorec_root not in sys.path:
    sys.path.insert(0, _duorec_root)

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from ncs_numpy_compat import apply_numpy_recbole_compat

apply_numpy_recbole_compat()

try:
    import pandas as pd

    pd.options.mode.copy_on_write = False
except ImportError:
    pass

from recbole.quick_start import run_recbole


def _smoke_data_path(dataset: str) -> str:
    """Tạo bản .inter nhỏ cho smoke test."""
    n = int(os.environ.get("NCS_SMOKE_SAMPLES", "5000"))
    base = Path(tempfile.mkdtemp(prefix="duorec_smoke_"))
    dst = base / dataset
    dst.mkdir(parents=True)
    src = Path(_duorec_root) / "dataset" / dataset / f"{dataset}.inter"
    lines = src.read_text(encoding="utf-8").splitlines()
    (dst / f"{dataset}.inter").write_text("\n".join(lines[: n + 1]) + "\n", encoding="utf-8")
    return str(base) + "/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="DuoRec", help="name of models")
    parser.add_argument("--dataset", "-d", type=str, default="ml-100k", help="name of datasets")
    parser.add_argument("--config_files", type=str, default="seq.yaml", help="config files")

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(" ") if args.config_files else None
    config_dict = None
    if os.environ.get("NCS_SMOKE"):
        config_dict = {
            "data_path": _smoke_data_path(args.dataset),
            "use_gpu": False,
            "gpu_id": "",
        }
    run_recbole(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
