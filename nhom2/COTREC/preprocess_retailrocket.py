#!/usr/bin/env python3
"""Chuyển RetailRocket -> Data/COTREC/retailrocket/"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
subprocess.run(
    [sys.executable, str(REPO / "scripts/preprocess_retailrocket.py"), "-m", "COTREC"],
    check=True,
)
