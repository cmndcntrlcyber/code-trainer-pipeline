#!/usr/bin/env bash
set -euo pipefail

# Bootstrap an edge device to sync Claude sessions to the central R2 bucket.
#
# Usage:
#   bash scripts/bootstrap_edge_sync.sh \
#     --api-token CF_API_TOKEN \
#     --account-id CF_ACCOUNT_ID \
#     --bucket cot-sessions \
#     --interval 30
#
# Requires: Node.js 18+ (for wrangler)

# ── Defaults ────────────────────────────────────────────────────────────────
BUCKET="cot-sessions"
INTERVAL=30
HOST_PREFIX="edge-$(hostname -s)"

# ── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-token)   API_TOKEN="$2"; shift 2 ;;
    --account-id)  ACCOUNT_ID="$2"; shift 2 ;;
    --bucket)      BUCKET="$2"; shift 2 ;;
    --interval)    INTERVAL="$2"; shift 2 ;;
    --host-prefix) HOST_PREFIX="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

for var in API_TOKEN ACCOUNT_ID; do
  if [ -z "${!var:-}" ]; then
    echo "ERROR: --$(echo "$var" | tr '_' '-' | tr '[:upper:]' '[:lower:]') is required"
    exit 1
  fi
done

log() { echo "[edge-bootstrap] $*"; }

# ── Step 1: Install wrangler if missing ─────────────────────────────────────
if ! command -v wrangler &>/dev/null; then
  if ! command -v npm &>/dev/null; then
    log "ERROR: npm not found. Install Node.js 18+ first."
    exit 1
  fi
  log "Installing wrangler..."
  npm install -g wrangler
else
  log "wrangler already installed: $(wrangler --version 2>/dev/null || echo 'unknown')"
fi

# ── Step 2: Write env vars to a persistent file ────────────────────────────
ENV_FILE="${HOME}/.config/claude-sync/env"
mkdir -p "$(dirname "$ENV_FILE")"

cat > "$ENV_FILE" <<EOF
CLOUDFLARE_API_TOKEN=${API_TOKEN}
CLOUDFLARE_ACCOUNT_ID=${ACCOUNT_ID}
R2_BUCKET=${BUCKET}
HOST_PREFIX=${HOST_PREFIX}
EOF

chmod 600 "$ENV_FILE"
log "Credentials saved to $ENV_FILE"

# ── Step 3: Write the sync script ──────────────────────────────────────────
SYNC_SCRIPT="${HOME}/.local/bin/sync-claude-sessions.sh"
mkdir -p "$(dirname "$SYNC_SCRIPT")"

cat > "$SYNC_SCRIPT" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${HOME}/.config/claude-sync/env"
LOG_FILE="${HOME}/.local/share/claude-sync/sync.log"
CLAUDE_PROJECTS_DIR="${HOME}/.claude/projects"
MARKER_DIR="${HOME}/.local/share/claude-sync/markers"

mkdir -p "$(dirname "$LOG_FILE")" "$MARKER_DIR"

if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
fi

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"; }

if [ ! -d "$CLAUDE_PROJECTS_DIR" ]; then
  log "No Claude projects directory, nothing to sync"
  exit 0
fi

if ! command -v wrangler &>/dev/null; then
  log "ERROR: wrangler not found"
  exit 1
fi

if [ -z "${CLOUDFLARE_API_TOKEN:-}" ]; then
  log "ERROR: CLOUDFLARE_API_TOKEN not set"
  exit 1
fi

if [ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]; then
  log "ERROR: CLOUDFLARE_ACCOUNT_ID not set"
  exit 1
fi

R2_BUCKET="${R2_BUCKET:-cot-sessions}"
HOST_PREFIX="${HOST_PREFIX:-edge-$(hostname -s)}"

declare -A ALLOWLIST=(
  ["*-ssd-training"]="training-ssd"
  ["*-code-nexus-harness"]="nexus-harness"
  ["*-code-rust-nexus"]="rust-nexus"
  ["*-code-rtpi"]="rtpi"
)

match_allowlist() {
  local dir_name="$1"
  for pattern in "${!ALLOWLIST[@]}"; do
    case "$dir_name" in
      $pattern) echo "${ALLOWLIST[$pattern]}"; return 0 ;;
    esac
  done
  return 1
}

uploaded=0
unchanged=0
skipped=0

for project_dir in "$CLAUDE_PROJECTS_DIR"/*/; do
  dir_name=$(basename "$project_dir")

  friendly_name=$(match_allowlist "$dir_name") || {
    skipped=$((skipped + 1))
    continue
  }

  while IFS= read -r -d '' file; do
    case "$file" in
      *.key|*/file-history/*|*/memory/*|*/session-env/*|*/shell-snapshots/*) continue ;;
    esac

    rel_path="${file#"$project_dir"}"
    r2_key="${HOST_PREFIX}/${friendly_name}/${rel_path}"

    marker_file="${MARKER_DIR}/$(echo "$r2_key" | md5sum | cut -d' ' -f1)"
    file_mtime=$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null)

    if [ -f "$marker_file" ] && [ "$(cat "$marker_file")" = "$file_mtime" ]; then
      unchanged=$((unchanged + 1))
      continue
    fi

    if (cd /tmp && wrangler r2 object put "${R2_BUCKET}/${r2_key}" --file "$file" --remote >/dev/null 2>&1); then
      echo "$file_mtime" > "$marker_file"
      uploaded=$((uploaded + 1))
    else
      log "  FAIL $r2_key"
    fi
  done < <(find "$project_dir" -type f -print0)
done

log "Sync: ${uploaded} uploaded, ${unchanged} unchanged, ${skipped} skipped (not in allowlist)"
SCRIPT

chmod +x "$SYNC_SCRIPT"
log "Wrote sync script to $SYNC_SCRIPT"

# ── Step 4: Install systemd timer or cron ───────────────────────────────────
if command -v systemctl &>/dev/null && systemctl --user status &>/dev/null; then
  UNIT_DIR="${HOME}/.config/systemd/user"
  mkdir -p "$UNIT_DIR"

  cat > "${UNIT_DIR}/claude-session-sync.service" <<EOF
[Unit]
Description=Sync Claude Code sessions to R2

[Service]
Type=oneshot
ExecStart=${SYNC_SCRIPT}
EnvironmentFile=${ENV_FILE}
Environment=PATH=/usr/local/bin:/usr/bin:/bin:%h/.local/bin:%h/.nvm/versions/node/current/bin
EOF

  cat > "${UNIT_DIR}/claude-session-sync.timer" <<EOF
[Unit]
Description=Sync Claude sessions every ${INTERVAL} minutes

[Timer]
OnCalendar=*:0/${INTERVAL}
Persistent=true

[Install]
WantedBy=timers.target
EOF

  systemctl --user daemon-reload
  systemctl --user enable --now claude-session-sync.timer
  log "Installed systemd timer (every ${INTERVAL}m)"
else
  CRON_MARKER="# claude-session-sync"
  CRON_ENTRY="*/${INTERVAL} * * * * . ${ENV_FILE} && ${SYNC_SCRIPT} >> ${HOME}/.local/share/claude-sync/sync.log 2>&1"
  (crontab -l 2>/dev/null | grep -v "$CRON_MARKER" || true; echo "${CRON_ENTRY} ${CRON_MARKER}") | crontab -
  log "Installed cron job (every ${INTERVAL}m)"
fi

# ── Step 5: Test and initial sync ───────────────────────────────────────────
export CLOUDFLARE_API_TOKEN="$API_TOKEN"
export CLOUDFLARE_ACCOUNT_ID="$ACCOUNT_ID"

log "Testing R2 connectivity..."
if (cd /tmp && wrangler r2 bucket list --remote >/dev/null 2>&1); then
  log "SUCCESS: R2 accessible"
else
  log "WARNING: Could not verify R2 access — check credentials"
fi

log "Running initial sync..."
"$SYNC_SCRIPT"

log ""
log "Edge device '${HOST_PREFIX}' is now syncing every ${INTERVAL}m."
log "  Manual sync:   ${SYNC_SCRIPT}"
log "  Timer status:  systemctl --user status claude-session-sync.timer"
log "  Logs:          tail -f ${HOME}/.local/share/claude-sync/sync.log"
