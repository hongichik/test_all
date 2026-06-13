#!/usr/bin/env python3
"""
Smoke test RetailRocket trên tất cả dự án — chạy lần lượt, ghi log từng project.

Log quá trình : Log/<project>/retailrocket/log-YYYY-MM-DD-HH-MM-SS.log
Log kết quả   : LogMins/<project>/retailrocket/DD-MM-YYYY.log
Tổng hợp      : Log/_smoke_all/retailrocket/log-YYYY-MM-DD-HH-MM-SS.log

Chạy ngầm từ thư mục gốc repo:
  nohup python3 scripts/run_all_retailrocket_smoke.py > /dev/null 2>&1 &
  # hoặc xem tiến trình:
  tail -f Log/_smoke_all/retailrocket/log-*.log
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ncs_logging import logmins_path, process_log_path, smoke_summary_log_path
from ncs_paths import REPO_ROOT
from scripts.retailrocket_papers import link_papers_datasets

PYTHON = sys.executable
DATASET = "retailrocket"
SMOKE_EPOCH = "1"
SMOKE_CONFIG = str(REPO / "config" / "smoke_1epoch.yaml")


@dataclass
class Job:
    project: str
    cwd: Path
    cmd: list[str]
    env: dict[str, str] | None = None


def _ensure_symlinks() -> None:
    pairs = (
        (REPO / "nhom3/DuoRec/dataset/retailrocket", REPO / "Data/DuoRec/retailrocket"),
        (REPO / "nhom3/CORE/dataset/retailrocket", REPO / "Data/CORE/retailrocket"),
    )
    for link, target in pairs:
        if not target.exists():
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            continue
        link.symlink_to(Path("../../../Data") / target.parent.name / target.name)


def _build_jobs() -> list[Job]:
    e1 = ["--dataset", DATASET, "--epoch", SMOKE_EPOCH]
    return [
        Job("SR-GNN", REPO / "nhom1/SR-GNN/pytorch_code", [PYTHON, "main.py", *e1]),
        Job("GCE-GNN", REPO / "nhom1/GCE-GNN", [PYTHON, "main.py", *e1]),
        Job("DHCN", REPO / "nhom2/DHCN", [PYTHON, "main.py", *e1]),
        Job("COTREC", REPO / "nhom2/COTREC", [PYTHON, "main.py", *e1]),
        Job("CSGNN", REPO / "nhom2/CSGNN", [PYTHON, "main.py", "--dataset", DATASET, "--epoch", SMOKE_EPOCH, "--embSize", "100", "--beta", "0.005"]),
        Job(
            "DuoRec",
            REPO / "nhom3/DuoRec",
            [PYTHON, "run_seq.py", "--dataset", DATASET, "--model", "DuoRec", "--config_files", f"seq.yaml DuoRec.yaml {SMOKE_CONFIG}"],
        ),
        Job(
            "CORE",
            REPO / "nhom3/CORE",
            [PYTHON, "main.py", "--model", "trm", "--dataset", DATASET],
            env={"NCS_SMOKE": "1", "CUDA_VISIBLE_DEVICES": ""},
        ),
        Job("SCL-DHCN", REPO / "nhom3/SCLRS/DHCN", [PYTHON, "main.py", *e1]),
        Job("SCL-COTREC", REPO / "nhom3/SCLRS/COTREC", [PYTHON, "main.py", *e1]),
        Job("SCL-GCE-GNN", REPO / "nhom3/SCLRS/GCE-GNN", [PYTHON, "main.py", *e1]),
        Job("FGNN", REPO / "papers_only/FGNN", [PYTHON, "main.py", *e1]),
        Job(
            "CM-HGNN",
            REPO / "papers_only/CM-HGNN",
            [PYTHON, "main.py", "--dataset", DATASET, "--epoch", SMOKE_EPOCH, "--batch_size", "32"],
            env={"CUDA_VISIBLE_DEVICES": ""},
        ),
        Job("CCT-GNN", REPO / "papers_only/CCT-GNN", [PYTHON, "main.py", *e1]),
        Job("HGCAN", REPO / "papers_only/HGCAN", [PYTHON, "main.py", "--dataset", DATASET, "--epochs", SMOKE_EPOCH]),
    ]


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def _run_job(job: Job) -> tuple[bool, str]:
    log_path = process_log_path(job.project, DATASET)
    env = os.environ.copy()
    env.setdefault("NCS_SMOKE", "1")
    env.setdefault("NCS_SMOKE_SAMPLES", "2000")
    if job.env:
        env.update(job.env)

    header = (
        f"{'=' * 72}\n"
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] START {job.project}\n"
        f"cwd: {job.cwd}\n"
        f"cmd: {' '.join(job.cmd)}\n"
        f"{'=' * 72}\n"
    )
    log_path.write_text(header, encoding="utf-8")

    t0 = time.time()
    try:
        with open(log_path, "a", encoding="utf-8") as logf:
            proc = subprocess.run(
                job.cmd,
                cwd=job.cwd,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        status = "OK" if ok else f"FAIL (exit {proc.returncode})"
        footer = f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] END {job.project} — {status} — {elapsed:.1f}s\n"
        _append(log_path, footer)

        mins = (
            f"[{datetime.now():%H:%M:%S}] {job.project}: {status} "
            f"({elapsed:.0f}s) -> {log_path.relative_to(REPO_ROOT)}\n"
        )
        _append(logmins_path(job.project, DATASET), mins)
        return ok, status
    except Exception as exc:
        elapsed = time.time() - t0
        err = f"\n[ERROR] {type(exc).__name__}: {exc}\n"
        _append(log_path, err)
        mins = f"[{datetime.now():%H:%M:%S}] {job.project}: ERROR {exc} ({elapsed:.0f}s)\n"
        _append(logmins_path(job.project, DATASET), mins)
        return False, str(exc)


def main() -> int:
    if Path.cwd().resolve() != REPO:
        os.chdir(REPO)

    summary_path = smoke_summary_log_path(DATASET)
    jobs = _build_jobs()
    _append(
        summary_path,
        f"Smoke RetailRocket — {len(jobs)} jobs — {datetime.now():%d-%m-%Y %H:%M:%S}\n",
    )

    _ensure_symlinks()
    try:
        link_papers_datasets()
    except OSError as exc:
        _append(summary_path, f"[warn] link papers_only: {exc}\n")

    ok_n, fail_n = 0, 0
    for i, job in enumerate(jobs, 1):
        line = f"\n[{i}/{len(jobs)}] {job.project} ...\n"
        print(line.strip())
        _append(summary_path, line)
        ok, status = _run_job(job)
        result = f"  -> {status}\n"
        print(result.strip())
        _append(summary_path, result)
        if ok:
            ok_n += 1
        else:
            fail_n += 1

    tail = f"\nDONE: {ok_n} OK, {fail_n} FAIL — summary: {summary_path}\n"
    print(tail.strip())
    _append(summary_path, tail)
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
