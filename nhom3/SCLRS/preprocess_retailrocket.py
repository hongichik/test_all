#!/usr/bin/env python3
"""Chuyển RetailRocket -> Data/SelfContrastiveLearningRecSys/retailrocket/{DHCN,COTREC,GCE-GNN}/"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for m in ("SCL-DHCN", "SCL-COTREC", "SCL-GCE-GNN"):
    print(f"\n=== {m} ===")
    subprocess.run(
        [sys.executable, str(REPO / "scripts/preprocess_retailrocket.py"), "-m", m],
        check=True,
    )
