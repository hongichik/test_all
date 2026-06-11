#!/usr/bin/env python3
"""
Tải RetailRocket từ Kaggle về Data/datagoc/Retailrocket/.

Auth tự động qua scripts/ncs_kaggle.py:
  - config/kaggle/access_token  hoặc  config/kaggle/kaggle.json
  - Sao chép từ ~/.kaggle/ nếu có

Chạy từ thư mục gốc repo:
  python3 scripts/download_retailrocket.py
  python3 scripts/download_retailrocket.py --force
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.ncs_kaggle import bootstrap_kaggle_auth

bootstrap_kaggle_auth()

import kagglehub

from ncs_paths import RAW_RETAILROCKET

KAGGLE_HANDLE = "retailrocket/ecommerce-dataset"

REQUIRED_FILES = (
    "category_tree.csv",
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
)


def _missing_files(dest: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (dest / name).is_file()]


def download_retailrocket(*, force: bool = False, dest: Path | None = None) -> Path:
    out = dest or RAW_RETAILROCKET
    out.mkdir(parents=True, exist_ok=True)

    missing = _missing_files(out)
    if not missing and not force:
        print(f"Đã có đủ {len(REQUIRED_FILES)} file tại {out}")
        print("Dùng --force để tải lại từ Kaggle.")
        return out

    if missing and not force:
        print(f"Thiếu {len(missing)} file: {', '.join(missing)}")

    print(f"Tải {KAGGLE_HANDLE} từ Kaggle ...")
    cache_path = Path(
        kagglehub.dataset_download(KAGGLE_HANDLE, force_download=force)
    )

    for name in REQUIRED_FILES:
        src = cache_path / name
        if not src.is_file():
            raise FileNotFoundError(
                f"Kaggle dataset thiếu {name}. Thư mục cache: {cache_path}"
            )
        dst = out / name
        print(f"  copy {name} -> {dst}")
        shutil.copy2(src, dst)

    print(f"Xong. Data gốc tại {out}")
    return out


def _verify_events(dest: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("Bỏ qua --verify (cần: pip install pandas)")
        return

    events = dest / "events.csv"
    df = pd.read_csv(events, nrows=5)
    print("events.csv (5 dòng đầu):")
    print(df.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tải RetailRocket Kaggle -> Data/datagoc/Retailrocket/"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tải lại dù đã có đủ file hoặc đã cache",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Đọc thử 5 dòng events.csv sau khi tải (cần pandas)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=f"Thư mục đích (mặc định: {RAW_RETAILROCKET})",
    )
    args = parser.parse_args()

    if Path.cwd().resolve() != REPO_ROOT:
        print(f"Lưu ý: nên chạy từ {REPO_ROOT}", file=sys.stderr)

    dest = download_retailrocket(force=args.force, dest=args.dest)
    if args.verify:
        _verify_events(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
