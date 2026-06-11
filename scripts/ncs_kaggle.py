"""
Bootstrap xác thực Kaggle cho toàn repo.

Ưu tiên:
  1. Biến môi trường KAGGLE_API_TOKEN / KAGGLE_USERNAME+KAGGLE_KEY
  2. config/kaggle/ trong repo (access_token hoặc kaggle.json)
  3. Sao chép từ ~/.kaggle/ vào config/kaggle/ (một lần)
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from ncs_paths import REPO_ROOT

PROJECT_KAGGLE_DIR = REPO_ROOT / "config" / "kaggle"
SYSTEM_KAGGLE_DIR = Path.home() / ".kaggle"


def _has_env_creds() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN", "").strip():
        return True
    return bool(
        os.environ.get("KAGGLE_USERNAME", "").strip()
        and os.environ.get("KAGGLE_KEY", "").strip()
    )


def _sync_from_system() -> None:
    """Copy credential từ ~/.kaggle vào config/kaggle/ nếu repo chưa có."""
    if not SYSTEM_KAGGLE_DIR.is_dir():
        return
    PROJECT_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("access_token", "kaggle.json"):
        src = SYSTEM_KAGGLE_DIR / name
        dst = PROJECT_KAGGLE_DIR / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"[kaggle] Đã copy {src} -> {dst}")


def _apply_project_creds() -> bool:
    PROJECT_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

    token_file = PROJECT_KAGGLE_DIR / "access_token"
    if token_file.is_file() and not os.environ.get("KAGGLE_API_TOKEN", "").strip():
        os.environ["KAGGLE_API_TOKEN"] = token_file.read_text(encoding="utf-8").strip()

    creds_file = PROJECT_KAGGLE_DIR / "kaggle.json"
    if creds_file.is_file():
        os.environ.setdefault("KAGGLE_CONFIG_DIR", str(PROJECT_KAGGLE_DIR))
        if not _has_env_creds() and not os.environ.get("KAGGLE_API_TOKEN", "").strip():
            try:
                data = json.loads(creds_file.read_text(encoding="utf-8"))
                user = data.get("username", "").strip()
                key = data.get("key", "").strip()
                if user and key:
                    os.environ.setdefault("KAGGLE_USERNAME", user)
                    os.environ.setdefault("KAGGLE_KEY", key)
            except json.JSONDecodeError:
                pass

    return _has_env_creds() or os.environ.get("KAGGLE_API_TOKEN", "").strip() != ""


def bootstrap_kaggle_auth() -> bool:
    """
    Chuẩn bị credential trước khi gọi kagglehub.
    Trả về True nếu có auth; False nếu không (dataset public vẫn tải được).
    """
    if _has_env_creds():
        return True

    _sync_from_system()
    if _apply_project_creds():
        return True

    if (PROJECT_KAGGLE_DIR / "kaggle.json").exists():
        os.environ.setdefault("KAGGLE_CONFIG_DIR", str(PROJECT_KAGGLE_DIR))
        return True

    print(
        "[kaggle] Chưa có API key (config/kaggle/ trống, ~/.kaggle/ không có). "
        "Dataset public retailrocket/ecommerce-dataset vẫn tải được."
    )
    return False
