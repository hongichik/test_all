#!/usr/bin/env bash
# Pull code từ Git; log local luôn thắng; log Colab (chỉ có trên remote) được giữ lại.
#
# Quy tắc:
#   - Code (nhom*, papers_only*, scripts, notebooks, …): theo Git
#   - Log/ LogMins/ do máy này chạy: luôn giữ bản trên disk, không bị pull ghi đè
#   - Log từ Colab (push lên Git): file mới trên remote vẫn được kéo về
#
# Colab push log: git add -f Log/...  (vì Log/ nằm trong .gitignore)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BACKUP="$(mktemp -d)"
cleanup() { rm -rf "$BACKUP"; }
trap cleanup EXIT

for dir in Log LogMins; do
  if [[ -d "$dir" ]]; then
    mkdir -p "$BACKUP/$dir"
    cp -a "$dir/." "$BACKUP/$dir/"
  fi
done

PULL_MODE="${NCS_GIT_PULL_MODE:-rebase}"
case "$PULL_MODE" in
  rebase)  git pull --rebase "$@" ;;
  merge)   git pull --no-rebase "$@" ;;
  ff-only) git pull --ff-only "$@" ;;
  *) echo "NCS_GIT_PULL_MODE không hợp lệ: $PULL_MODE (rebase|merge|ff-only)" >&2; exit 1 ;;
esac

for dir in Log LogMins; do
  if [[ -d "$BACKUP/$dir" ]]; then
    mkdir -p "$dir"
    rsync -a "$BACKUP/$dir/" "$dir/"
  fi
done

echo "ncs_git_pull: xong — code theo Git, log local đã giữ nguyên (không dùng skip-worktree)."
