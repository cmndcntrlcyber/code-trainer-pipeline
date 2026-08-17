#!/usr/bin/env bash
set -euo pipefail

# Pull all edge device sessions from R2 into data/cot_rl_sessions/.
# Run on the training host — sessions are gitignored, never committed.
#
# Features:
#   - Paginated listing (handles >1000 objects)
#   - ETag-based change detection (re-downloads updated JSONL files)
#   - Full exclusion filtering (shared with upload scripts)
#
# Usage:
#   bash scripts/pull_sessions_from_r2.sh

# ── Shared config ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/sync_config.sh"

DEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/data/cot_rl_sessions"
MANIFEST="${DEST_DIR}/.r2_manifest.json"

log() { echo "[pull-sessions] $*"; }

# ── Validation ───────────────────────────────────────────────────────────────

require_deps wrangler curl python3
validate_cf_env || exit 1

mkdir -p "$DEST_DIR"
[ -f "$MANIFEST" ] || echo '{}' > "$MANIFEST"

# ── Paginated listing ────────────────────────────────────────────────────────

log "Listing objects in r2:${R2_BUCKET}..."

objects_file=$(mktemp)
trap 'rm -f "$objects_file"' EXIT

cursor=""
page=0
while true; do
  page=$((page + 1))
  url="${CF_API_BASE}/accounts/${CLOUDFLARE_ACCOUNT_ID}/r2/buckets/${R2_BUCKET}/objects?per_page=1000"
  [ -n "$cursor" ] && url="${url}&cursor=${cursor}"

  response=$(curl -sf \
    -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
    "$url") || { log "ERROR: R2 API request failed on page $page"; exit 1; }

  python3 -c "
import json, sys
data = json.loads(sys.argv[1])
for obj in data.get('result', []):
    print(json.dumps({'key': obj['key'], 'etag': obj.get('etag', ''), 'size': obj.get('size', 0)}))
" "$response" >> "$objects_file"

  cursor=$(python3 -c "
import json, sys
data = json.loads(sys.argv[1])
cursor = data.get('result_info', {}).get('cursor', '') or ''
print(cursor)
" "$response")

  [ -z "$cursor" ] && break
  log "  Page $page fetched, continuing..."
done

total=$(wc -l < "$objects_file")
if [ "$total" -eq 0 ]; then
  log "No objects found in bucket"
  exit 0
fi

log "Found $total objects across $page page(s)"

# ── Download with ETag change detection ──────────────────────────────────────

downloaded=0
skipped=0
failed=0
updated=0

while IFS= read -r obj_json; do
  key=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['key'])" "$obj_json")
  etag=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['etag'])" "$obj_json")
  size=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['size'])" "$obj_json")

  if should_exclude "$key"; then
    continue
  fi

  cached_etag=$(python3 -c "
import json, sys
m = json.load(open(sys.argv[1]))
print(m.get(sys.argv[2], {}).get('etag', ''))
" "$MANIFEST" "$key")

  if [ "$cached_etag" = "$etag" ] && [ -n "$etag" ]; then
    skipped=$((skipped + 1))
    continue
  fi

  dest_file="${DEST_DIR}/${key}"
  mkdir -p "$(dirname "$dest_file")"

  is_update="false"
  [ -f "$dest_file" ] && is_update="true"

  if (cd /tmp && wrangler r2 object get "${R2_BUCKET}/${key}" --file "$dest_file" --remote >/dev/null 2>&1); then
    python3 -c "
import json, sys
manifest_path, key, etag, size = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
m = json.load(open(manifest_path))
m[key] = {'etag': etag, 'size': int(size)}
json.dump(m, open(manifest_path, 'w'), indent=2)
" "$MANIFEST" "$key" "$etag" "$size"

    if [ "$is_update" = "true" ]; then
      updated=$((updated + 1))
      log "  UPDATED $key"
    else
      downloaded=$((downloaded + 1))
    fi
  else
    log "  FAIL $key"
    failed=$((failed + 1))
  fi
done < "$objects_file"

hosts=$(find "$DEST_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
sessions=$(find "$DEST_DIR" -name "*.jsonl" 2>/dev/null | wc -l)
disk_size=$(du -sh "$DEST_DIR" 2>/dev/null | cut -f1)

log "Done: ${downloaded} new, ${updated} updated, ${skipped} unchanged, ${failed} failed"
log "Total: ${sessions} sessions from ${hosts} hosts (${disk_size})"
