#!/usr/bin/env bash
# Shared config for the CoT/RL session sync pipeline.
# Sourced by sync_sessions_to_r2.sh, pull_sessions_from_r2.sh, and others.
# Do not execute directly.

# ── Source .env as fallback (shell env vars take precedence) ─────────────────
_SYNC_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${_SYNC_CONFIG_DIR}/.env" ]; then
  while IFS='=' read -r key value; do
    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
    value="${value%"${value##*[![:space:]]}"}"
    [ -z "${!key:-}" ] && export "$key"="$value"
  done < "${_SYNC_CONFIG_DIR}/.env"
fi

R2_BUCKET="${R2_BUCKET:-cot-sessions}"
CF_API_BASE="https://api.cloudflare.com/client/v4"

# ── Allowlist: glob pattern -> friendly name ────────────────────────────────
declare -A ALLOWLIST=(
  ["*-ssd-training"]="training-ssd"
  ["*-code-nexus-harness"]="nexus-harness"
  ["*-code-rust-nexus"]="rust-nexus"
  ["*-code-rtpi"]="rtpi"
)

# ── Exclusion filter ────────────────────────────────────────────────────────
should_exclude() {
  local path="$1"
  case "$path" in
    *.key) return 0 ;;
    */file-history/*) return 0 ;;
    */memory/*) return 0 ;;
    */session-env/*) return 0 ;;
    */shell-snapshots/*) return 0 ;;
  esac
  return 1
}

# ── Allowlist matching ──────────────────────────────────────────────────────
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

# ── Environment validation ──────────────────────────────────────────────────
validate_cf_env() {
  local ok=true
  if [ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]; then
    echo "ERROR: CLOUDFLARE_ACCOUNT_ID not set" >&2
    ok=false
  fi
  if [ -z "${CLOUDFLARE_API_TOKEN:-}" ]; then
    echo "ERROR: CLOUDFLARE_API_TOKEN not set" >&2
    ok=false
  fi
  $ok || return 1
}

# ── Dependency check ────────────────────────────────────────────────────────
require_deps() {
  local missing=()
  for cmd in "$@"; do
    command -v "$cmd" &>/dev/null || missing+=("$cmd")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: Missing required commands: ${missing[*]}" >&2
    return 1
  fi
}
