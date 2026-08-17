#!/usr/bin/env bash
set -euo pipefail

# ── Shared config ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/sync_config.sh"

HOST_PREFIX="${HOST_PREFIX:-local}"
CLAUDE_PROJECTS_DIR="${HOME}/.claude/projects"

# ── Functions ────────────────────────────────────────────────────────────────

log() { echo "[sync-sessions] $*"; }

upload_file() {
  local src="$1" r2_key="$2" attempt=0 max_retries=3
  while [ $attempt -lt $max_retries ]; do
    if (cd /tmp && wrangler r2 object put "${R2_BUCKET}/${r2_key}" --file "$src" --remote >/dev/null 2>&1); then
      return 0
    fi
    attempt=$((attempt + 1))
    [ $attempt -lt $max_retries ] && sleep 2
  done
  log "  FAIL $r2_key (after $max_retries attempts)"
  return 1
}

# ── Validation ───────────────────────────────────────────────────────────────

require_deps wrangler
validate_cf_env || exit 1

if [ ! -d "$CLAUDE_PROJECTS_DIR" ]; then
  log "ERROR: Claude projects directory not found at $CLAUDE_PROJECTS_DIR"
  exit 1
fi

# ── Main ─────────────────────────────────────────────────────────────────────

MARKER_DIR="${HOME}/.local/share/claude-sync/markers"
mkdir -p "$MARKER_DIR"

synced=0
uploaded=0
unchanged=0
failed=0
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

  log "SYNC $friendly_name ($session_count sessions)"

  while IFS= read -r -d '' file; do
    if should_exclude "$file"; then
      continue
    fi

    rel_path="${file#"$project_dir"}"
    r2_key="${HOST_PREFIX}/${friendly_name}/${rel_path}"

    marker_file="${MARKER_DIR}/$(echo "$r2_key" | md5sum | cut -d' ' -f1)"
    file_mtime=$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null)

    if [ -f "$marker_file" ] && [ "$(cat "$marker_file")" = "$file_mtime" ]; then
      unchanged=$((unchanged + 1))
      continue
    fi

    log "  PUT $rel_path"
    if upload_file "$file" "$r2_key"; then
      echo "$file_mtime" > "$marker_file"
      uploaded=$((uploaded + 1))
    else
      failed=$((failed + 1))
    fi
  done < <(find "$project_dir" -type f -print0)

  synced=$((synced + 1))
done

log "Done: $synced projects, $uploaded uploaded, $unchanged unchanged, $failed failed, $skipped skipped"
[ "$failed" -gt 0 ] && exit 1 || exit 0
