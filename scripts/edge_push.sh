#!/usr/bin/env bash
set -euo pipefail

# Self-contained edge session pusher for Kali Docker containers.
# No repo, no Node.js, no wrangler — just curl + bash.
#
# Pushes Claude Code session .jsonl files from ~/.claude/projects/
# to Cloudflare R2 via the S3-compatible API.
#
# Usage (one-shot):
#   CLOUDFLARE_API_TOKEN=xxx CLOUDFLARE_ACCOUNT_ID=yyy bash edge_push.sh
#
# Usage (install cron for recurring push every 30 min):
#   CLOUDFLARE_API_TOKEN=xxx CLOUDFLARE_ACCOUNT_ID=yyy bash edge_push.sh --install
#
# Requires: bash, curl, openssl (all standard on Kali)

BUCKET="${R2_BUCKET:-cot-sessions}"
HOST_PREFIX="edge-$(hostname -s)"
CLAUDE_DIR="${HOME}/.claude/projects"
MARKER_DIR="${HOME}/.local/share/claude-sync/markers"
LOG_FILE="${HOME}/.local/share/claude-sync/sync.log"

# ── Validation ──────────────────────────────────────────────────────────────

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"; }

[ -z "${CLOUDFLARE_API_TOKEN:-}" ] && die "CLOUDFLARE_API_TOKEN not set"
[ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ] && die "CLOUDFLARE_ACCOUNT_ID not set"
command -v curl &>/dev/null || die "curl not found"

mkdir -p "$MARKER_DIR" "$(dirname "$LOG_FILE")"

# ── Install mode: write creds + add cron ────────────────────────────────────

if [ "${1:-}" = "--install" ]; then
  ENV_FILE="${HOME}/.config/claude-sync/env"
  mkdir -p "$(dirname "$ENV_FILE")"
  cat > "$ENV_FILE" <<EOF
CLOUDFLARE_API_TOKEN=${CLOUDFLARE_API_TOKEN}
CLOUDFLARE_ACCOUNT_ID=${CLOUDFLARE_ACCOUNT_ID}
R2_BUCKET=${BUCKET}
HOST_PREFIX=${HOST_PREFIX}
EOF
  chmod 600 "$ENV_FILE"

  SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  if [ ! -f "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="${HOME}/.local/bin/edge_push.sh"
    cp "$0" "$SCRIPT_PATH"
    chmod +x "$SCRIPT_PATH"
  fi

  CRON_MARKER="# claude-edge-push"
  CRON_ENTRY="*/30 * * * * . ${ENV_FILE} && bash ${SCRIPT_PATH} >> ${LOG_FILE} 2>&1"
  (crontab -l 2>/dev/null | grep -v "$CRON_MARKER" || true; echo "${CRON_ENTRY} ${CRON_MARKER}") | crontab -

  log "Installed cron (every 30m). Creds at ${ENV_FILE}. Running initial push..."
fi

# ── Exclusion filter ────────────────────────────────────────────────────────

should_exclude() {
  case "$1" in
    *.key|*/file-history/*|*/memory/*|*/session-env/*|*/shell-snapshots/*) return 0 ;;
  esac
  return 1
}

# ── Upload via Cloudflare API ───────────────────────────────────────────────

upload_file() {
  local src="$1" r2_key="$2" attempt=0
  local url="https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/r2/buckets/${BUCKET}/objects/${r2_key}"

  while [ $attempt -lt 3 ]; do
    if curl -sf -X PUT "$url" \
        -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
        -H "Content-Type: application/octet-stream" \
        --data-binary @"$src" >/dev/null 2>&1; then
      return 0
    fi
    attempt=$((attempt + 1))
    [ $attempt -lt 3 ] && sleep 2
  done
  return 1
}

# ── Main ────────────────────────────────────────────────────────────────────

if [ ! -d "$CLAUDE_DIR" ]; then
  log "No Claude projects at ${CLAUDE_DIR}, nothing to push"
  exit 0
fi

uploaded=0
unchanged=0
failed=0

for project_dir in "$CLAUDE_DIR"/*/; do
  [ -d "$project_dir" ] || continue
  dir_name=$(basename "$project_dir")

  session_count=$(find "$project_dir" -maxdepth 1 -name "*.jsonl" 2>/dev/null | wc -l)
  [ "$session_count" -eq 0 ] && continue

  while IFS= read -r -d '' file; do
    should_exclude "$file" && continue

    rel_path="${file#"$project_dir"}"
    r2_key="${HOST_PREFIX}/${dir_name}/${rel_path}"

    marker_file="${MARKER_DIR}/$(echo "$r2_key" | md5sum | cut -d' ' -f1)"
    file_mtime=$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null)

    if [ -f "$marker_file" ] && [ "$(cat "$marker_file")" = "$file_mtime" ]; then
      unchanged=$((unchanged + 1))
      continue
    fi

    if upload_file "$file" "$r2_key"; then
      echo "$file_mtime" > "$marker_file"
      uploaded=$((uploaded + 1))
    else
      log "  FAIL $r2_key"
      failed=$((failed + 1))
    fi
  done < <(find "$project_dir" -type f -print0)
done

log "Push: ${uploaded} uploaded, ${unchanged} unchanged, ${failed} failed"
[ "$failed" -gt 0 ] && exit 1 || exit 0
