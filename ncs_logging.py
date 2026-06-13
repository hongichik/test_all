"""Helper đường dẫn log theo quy ước NCS."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ncs_paths import REPO_ROOT

LOG_ROOT = REPO_ROOT / "Log"
LOGMINS_ROOT = REPO_ROOT / "LogMins"

# ISO-like: sort A→Z = cũ→mới (file mới nằm dưới cùng trong file explorer / ls)
PROCESS_LOG_TIME_FMT = "%Y-%m-%d-%H-%M-%S"


def process_log_filename(now: datetime | None = None) -> str:
    """Tên file log quá trình: log-YYYY-MM-DD-HH-MM-SS.log"""
    ts = (now or datetime.now()).strftime(PROCESS_LOG_TIME_FMT)
    return f"log-{ts}.log"


def process_log_path(project: str, dataset: str = "retailrocket") -> Path:
    """Log quá trình: Log/<project>/<dataset>/log-YYYY-MM-DD-HH-MM-SS.log"""
    d = LOG_ROOT / project / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / process_log_filename()


def logmins_path(project: str, dataset: str = "retailrocket") -> Path:
    """Log kết quả: LogMins/<project>/<dataset>/DD-MM-YYYY.log"""
    now = datetime.now()
    d = LOGMINS_ROOT / project / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{now.strftime('%d-%m-%Y')}.log"


def smoke_summary_log_path(dataset: str = "retailrocket") -> Path:
    """Tổng hợp smoke test tất cả dự án."""
    d = LOG_ROOT / "_smoke_all" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / process_log_filename()
