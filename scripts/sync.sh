#!/usr/bin/env bash
# Synchronize this repo with a remote cluster over rsync + ssh.
#
#   scripts/sync.sh push [--dry-run]                     copy the repo to the cluster
#   scripts/sync.sh pull REMOTE_PATH [LOCAL_PATH] [--dry-run]
#                                                        copy a path under the remote repo back
#
# SYNC_HOST (default access-hpc3.rcic.uci.edu) and SYNC_REMOTE_DIR (default
# /pub/$USER/boiling-data) choose where the repo lives on the cluster.
set -euo pipefail

host="${SYNC_HOST:-afeeney@access-hpc3.rcic.uci.edu}"
remote_dir="${SYNC_REMOTE_DIR:-/pub/${USER}/project/boiling-data}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# .git is left out so a clone on the cluster keeps its own history; the
# environment and caches are rebuilt there by uv
excludes=(
    --exclude .git/
    --exclude .venv/
    --exclude __pycache__/
    --exclude .pytest_cache/
    --exclude .mypy_cache/
    --exclude .ruff_cache/
    --exclude .DS_Store
)

usage() {
    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
}

command="${1:-}"
[[ -n "${command}" ]] || usage
shift

rsync_options=(--archive --compress --human-readable --progress)
positional=()
for argument in "$@"; do
    case "${argument}" in
        --dry-run) rsync_options+=(--dry-run) ;;
        *) positional+=("${argument}") ;;
    esac
done

case "${command}" in
    push)
        [[ ${#positional[@]} -eq 0 ]] || usage
        # the transfer node rejects plain ssh commands like mkdir, so rsync creates
        # the repo directory itself; it can only create that last level, so its
        # parent must already exist (macOS's openrsync has no --mkpath)
        rsync "${rsync_options[@]}" "${excludes[@]}" "${repo_dir}/" "${host}:${remote_dir}/"
        ;;
    pull)
        [[ ${#positional[@]} -ge 1 && ${#positional[@]} -le 2 ]] || usage
        remote_path="${positional[0]}"
        local_path="${positional[1]:-${repo_dir}/${remote_path}}"
        mkdir -p "$(dirname "${local_path}")"
        rsync "${rsync_options[@]}" "${host}:${remote_dir}/${remote_path}" "${local_path}"
        ;;
    *)
        usage
        ;;
esac
