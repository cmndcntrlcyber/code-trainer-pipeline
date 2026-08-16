#!/usr/bin/env bash
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
R2_REMOTE="${R2_REMOTE:-r2}"
R2_BUCKET="${R2_BUCKET:-cot-sessions}"
HOST_PREFIX="${HOST_PREFIX:-local}"
CLAUDE_PROJECTS_DIR="${HOME}/.claude/projects"

declare -A ALLOWLIST=(
  ["*-ssd-training"]="training-ssd"
  ["*-code-nexus-harness"]="nexus-harness"
  ["*-code-rust-nexus"]="rust-nexus"
  ["*-code-rtpi"]="rtpi"
)

EXCLUDE_PATTERNS=(
  "*.key"
  "file-history/**"
  "memory/**"
  "session-env/**"
  "shell-snapshots/**"
)

# ── Functions ────────────────────────────────────────────────────────────────

log() { echo "[sync-sessions] $*"; }

match_allowlist() {
  local dir_name="$1"
  for pattern in "${!ALLOWLIST[@]}"; do
    # shellcheck disable=SC2254
    case "$dir_name" in
      $pattern) echo "${ALLOWLIST[$pattern]}"; return 0 ;;
    esac
  done
  return 1
}

build_exclude_flags() {
  local flags=()
  for pat in "${EXCLUDE_PATTERNS[@]}"; do
    flags+=(--exclude "$pat")
  done
  echo "${flags[@]}"
}

# ── Main ─────────────────────────────────────────────────────────────────────

if ! command -v rclone &>/dev/null; then
  log "ERROR: rclone is not installed. Install it: https://rclone.org/install/"
  exit 1
fi

if ! rclone listremotes | grep -q "^${R2_REMOTE}:$"; then
  log "ERROR: rclone remote '${R2_REMOTE}' not found. Run: rclone config"
  exit 1
fi

if [ ! -d "$CLAUDE_PROJECTS_DIR" ]; then
  log "ERROR: Claude projects directory not found at $CLAUDE_PROJECTS_DIR"
  exit 1
fi

synced=0
skipped=0

for project_dir in "$CLAUDE_PROJECTS_DIR"/*/; do
  dir_name=$(basename "$project_dir")

  friendly_name=$(match_allowlist "$dir_name") || {
    skipped=$((skipped + 1))
    continue
  }

  session_count=$(find "$project_dir" -maxdepth 1 -name "*.jsonl" 2>/dev/null | wc -l)
  if [ "$session_count" -eq 0 ]; then
    log "SKIP $friendly_name (no sessions)"
    continue
  fi

  dest="${R2_REMOTE}:${R2_BUCKET}/${HOST_PREFIX}/${friendly_name}/"
  log "SYNC $friendly_name ($session_count sessions) -> $dest"

  # shellcheck disable=SC2046
  rclone sync "$project_dir" "$dest" \
    --update \
    $(build_exclude_flags) \
    --transfers 4 \
    --checkers 8 \
    --stats-one-line \
    -v

  synced=$((synced + 1))
done

log "Done: $synced projects synced, $skipped skipped (not in allowlist)"
