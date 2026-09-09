#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  sync_and_prepare.sh — End-to-end data sync & preparation      ║
# ║                                                                  ║
# ║  Default (fast): R2 pull → ingest → prompts → FARCA precompute  ║
# ║  --refresh-corpus: + offsec repo scrape → DAPT corpus build     ║
# ║  --full: everything above combined                               ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Usage:
#   # Fast path (R2 sessions + RL data prep only):
#   bash scripts/sync_and_prepare.sh
#
#   # Include offsec corpus refresh (GitHub scrape + DAPT corpus):
#   bash scripts/sync_and_prepare.sh --refresh-corpus
#
#   # Full pipeline (corpus + sessions + RL data + FARCA):
#   bash scripts/sync_and_prepare.sh --full
#
#   # Skip R2 pull (local data already fresh):
#   bash scripts/sync_and_prepare.sh --skip-pull
#
#   # Push datasets to Hub after preparation:
#   bash scripts/sync_and_prepare.sh --push-to-hub
#
#   # Dry run (show what would happen):
#   bash scripts/sync_and_prepare.sh --dry-run
#
#   # Install as systemd timer (runs every 2 hours):
#   bash scripts/sync_and_prepare.sh --install-timer
#
#   # Install as cron job (runs every 2 hours):
#   bash scripts/sync_and_prepare.sh --install-cron
#
# Required env vars (set in .env or shell):
#   CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN  — for R2 pull
#   GITHUB_TOKEN                                 — for offsec corpus refresh
#   HF_TOKEN or HUGGINGFACE_TOKEN                — for Hub push (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ─── Load .env if present ────────────────────────────────────────
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ─── Defaults ─────────────────────────────────────────────────────
SKIP_PULL=false
PUSH_TO_HUB=false
DRY_RUN=false
INSTALL_TIMER=false
INSTALL_CRON=false
REFRESH_CORPUS=false
FULL=false
LOG_DIR="$PROJECT_ROOT/data/sync_logs"
LOCK_FILE="/tmp/sync_and_prepare.lock"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-src/config/config.yaml}"
OFFSEC_CONFIG="${OFFSEC_CONFIG:-src/config/offsec_config.yaml}"
MAX_PROMPTS="${MAX_PROMPTS:-1000}"

# ─── Parse args ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-pull)        SKIP_PULL=true; shift ;;
        --push-to-hub)      PUSH_TO_HUB=true; shift ;;
        --dry-run)          DRY_RUN=true; shift ;;
        --install-timer)    INSTALL_TIMER=true; shift ;;
        --install-cron)     INSTALL_CRON=true; shift ;;
        --refresh-corpus)   REFRESH_CORPUS=true; shift ;;
        --full)             FULL=true; REFRESH_CORPUS=true; shift ;;
        --config)           PIPELINE_CONFIG="$2"; shift 2 ;;
        --offsec-config)    OFFSEC_CONFIG="$2"; shift 2 ;;
        --max-prompts)      MAX_PROMPTS="$2"; shift 2 ;;
        *)                  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Timer/cron installation ─────────────────────────────────────
if $INSTALL_TIMER; then
    echo "Installing systemd user timer for sync_and_prepare..."
    mkdir -p "$HOME/.config/systemd/user" "$LOG_DIR"

    cat > "$HOME/.config/systemd/user/farca-data-sync.service" <<UNIT
[Unit]
Description=FARCA Data Sync and Preparation
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=$PROJECT_ROOT
ExecStart=$PROJECT_ROOT/scripts/sync_and_prepare.sh
StandardOutput=append:$LOG_DIR/sync.log
StandardError=append:$LOG_DIR/sync.log
Environment="PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
UNIT

    cat > "$HOME/.config/systemd/user/farca-data-sync.timer" <<UNIT
[Unit]
Description=Run FARCA data sync every 2 hours

[Timer]
OnBootSec=10min
OnUnitActiveSec=2h
Persistent=true

[Install]
WantedBy=timers.target
UNIT

    systemctl --user daemon-reload
    systemctl --user enable --now farca-data-sync.timer
    echo "Timer installed and started. Check with: systemctl --user status farca-data-sync.timer"
    exit 0
fi

if $INSTALL_CRON; then
    echo "Installing cron job for sync_and_prepare (every 2 hours)..."
    mkdir -p "$LOG_DIR"
    CRON_CMD="0 */2 * * * cd $PROJECT_ROOT && bash scripts/sync_and_prepare.sh >> $LOG_DIR/sync.log 2>&1"
    (crontab -l 2>/dev/null | grep -v "sync_and_prepare"; echo "$CRON_CMD") | crontab -
    echo "Cron job installed. Check with: crontab -l"
    exit 0
fi

# ─── Locking (prevent concurrent runs) ───────────────────────────
if [[ -f "$LOCK_FILE" ]]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "[$(date -Iseconds)] sync_and_prepare already running (PID $LOCK_PID), skipping."
        exit 0
    fi
    rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# ─── Logging ──────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$RUN_LOG"
}

# Count total steps
TOTAL_STEPS=4
if $REFRESH_CORPUS; then
    TOTAL_STEPS=6
fi

log "=========================================="
log "FARCA Data Sync & Prepare Pipeline"
log "  project_root:    $PROJECT_ROOT"
log "  pipeline_config: $PIPELINE_CONFIG"
log "  offsec_config:   $OFFSEC_CONFIG"
log "  refresh_corpus:  $REFRESH_CORPUS"
log "  skip_pull:       $SKIP_PULL"
log "  push_to_hub:     $PUSH_TO_HUB"
log "  dry_run:         $DRY_RUN"
log "  total_steps:     $TOTAL_STEPS"
log "=========================================="

if $DRY_RUN; then
    STEP=1
    if $REFRESH_CORPUS; then
        log "[DRY RUN] Step $STEP/$TOTAL_STEPS: Scrape offsec repos (run_offsec_collection.py --skip-capture)"
        ((STEP++))
        log "[DRY RUN] Step $STEP/$TOTAL_STEPS: Build DAPT corpus (prepare_corpus.py)"
        ((STEP++))
    fi
    log "[DRY RUN] Step $STEP/$TOTAL_STEPS: Pull sessions from R2 (pull_sessions_from_r2.sh)"
    ((STEP++))
    log "[DRY RUN] Step $STEP/$TOTAL_STEPS: Ingest OCO sessions (ingest_oco_sessions.py)"
    ((STEP++))
    log "[DRY RUN] Step $STEP/$TOTAL_STEPS: Build GRPO prompts (build_grpo_prompts.py)"
    ((STEP++))
    log "[DRY RUN] Step $STEP/$TOTAL_STEPS: Pre-compute FARCA annotations (precompute_farca_annotations.py)"
    if $PUSH_TO_HUB; then
        log "[DRY RUN]   + Push datasets to HuggingFace Hub"
    fi
    exit 0
fi

CHANGED=false
STEP=0

# ═══════════════════════════════════════════════════════════════════
# PHASE A: OFFSEC CORPUS (optional, --refresh-corpus or --full)
# ═══════════════════════════════════════════════════════════════════

if $REFRESH_CORPUS; then

    # ─── Step: Scrape offsec repos ────────────────────────────────
    ((STEP++))
    log "Step $STEP/$TOTAL_STEPS: Scraping offensive security repositories..."
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        log "  WARNING: GITHUB_TOKEN not set, skipping offsec scrape"
    elif [[ ! -f "$OFFSEC_CONFIG" ]]; then
        log "  WARNING: offsec config not found at $OFFSEC_CONFIG, skipping"
    else
        REPOS_BEFORE=$(find data/offensive-security/repositories -maxdepth 1 -type d 2>/dev/null | wc -l)

        uv run python -m src.phase1_data_collection.scripts.run_offsec_collection \
            --config "$OFFSEC_CONFIG" \
            --skip-capture \
            >> "$RUN_LOG" 2>&1 || {
            log "  WARNING: Offsec scrape failed (non-fatal)"
        }

        REPOS_AFTER=$(find data/offensive-security/repositories -maxdepth 1 -type d 2>/dev/null | wc -l)
        NEW_REPOS=$((REPOS_AFTER - REPOS_BEFORE))
        if [[ $NEW_REPOS -gt 0 ]]; then
            CHANGED=true
            log "  $NEW_REPOS new repos cloned (total: $REPOS_AFTER)"
        else
            log "  No new repos (total: $REPOS_AFTER)"
        fi
    fi

    # ─── Step: Build DAPT corpus ──────────────────────────────────
    ((STEP++))
    log "Step $STEP/$TOTAL_STEPS: Building DAPT corpus from offsec repositories..."
    CORPUS_ARGS=(
        --config "$PIPELINE_CONFIG"
        --output-dir data/dapt_corpus
    )
    if $PUSH_TO_HUB; then
        CORPUS_ARGS+=(--push-to-hub)
    fi

    uv run python -m src.phase3b_dapt.data.prepare_corpus \
        "${CORPUS_ARGS[@]}" \
        >> "$RUN_LOG" 2>&1 || {
        log "  WARNING: DAPT corpus preparation failed (non-fatal)"
    }

    if [[ -d "data/dapt_corpus" ]]; then
        CORPUS_FILES=$(find data/dapt_corpus -name "*.arrow" -o -name "*.jsonl" 2>/dev/null | wc -l)
        log "  DAPT corpus files: $CORPUS_FILES"
    fi
fi

# ═══════════════════════════════════════════════════════════════════
# PHASE B: R2 SESSION SYNC + RL DATA PREP
# ═══════════════════════════════════════════════════════════════════

# ─── Step: Pull sessions from R2 ─────────────────────────────────
((STEP++))
if ! $SKIP_PULL; then
    log "Step $STEP/$TOTAL_STEPS: Pulling sessions from R2..."
    if [[ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]] || [[ -z "${CLOUDFLARE_API_TOKEN:-}" ]]; then
        log "  WARNING: Cloudflare credentials not set, skipping R2 pull"
    else
        PULL_OUTPUT=$(bash scripts/pull_sessions_from_r2.sh 2>&1) || true
        echo "$PULL_OUTPUT" >> "$RUN_LOG"

        if echo "$PULL_OUTPUT" | grep -qiE "downloaded|new|updated|changed"; then
            CHANGED=true
            log "  New/updated sessions detected"
        else
            log "  No new sessions from R2"
        fi
    fi
else
    log "Step $STEP/$TOTAL_STEPS: Skipping R2 pull (--skip-pull)"
    CHANGED=true
fi

# ─── Step: Ingest OCO sessions ───────────────────────────────────
((STEP++))
log "Step $STEP/$TOTAL_STEPS: Ingesting OCO sessions..."
INGEST_BEFORE=""
if [[ -f "data/oco_converted/ingestion_stats.json" ]]; then
    INGEST_BEFORE=$(cat data/oco_converted/ingestion_stats.json)
fi

uv run python -m src.phase4c_rl.data.ingest_oco_sessions \
    --input-dir data/cot_rl_sessions \
    --output-dir data/oco_converted \
    --format json \
    >> "$RUN_LOG" 2>&1 || {
    log "  WARNING: Ingestion failed (non-fatal)"
}

if [[ -f "data/oco_converted/ingestion_stats.json" ]]; then
    INGEST_AFTER=$(cat data/oco_converted/ingestion_stats.json)
    if [[ "$INGEST_BEFORE" != "$INGEST_AFTER" ]]; then
        CHANGED=true
        log "  Ingestion stats changed — new data processed"
    else
        log "  Ingestion stats unchanged"
    fi

    CONVERTED=$(python3 -c "import json; print(json.load(open('data/oco_converted/ingestion_stats.json')).get('converted', 0))" 2>/dev/null || echo "?")
    log "  Converted sessions: $CONVERTED"
fi

# ─── Step: Build GRPO prompts ────────────────────────────────────
((STEP++))
log "Step $STEP/$TOTAL_STEPS: Building GRPO prompts..."
PROMPT_ARGS=(
    --config "$PIPELINE_CONFIG"
    --output-dir data/grpo_prompts
    --max-prompts "$MAX_PROMPTS"
)
if $PUSH_TO_HUB; then
    PROMPT_ARGS+=(--push-to-hub)
fi

uv run python -m src.phase4c_rl.data.build_grpo_prompts \
    "${PROMPT_ARGS[@]}" \
    >> "$RUN_LOG" 2>&1 || {
    log "  WARNING: Prompt building failed (non-fatal)"
}

if [[ -f "data/grpo_prompts/prompt_stats.json" ]]; then
    PROMPT_COUNT=$(python3 -c "import json; print(json.load(open('data/grpo_prompts/prompt_stats.json')).get('total', 0))" 2>/dev/null || echo "?")
    log "  Built $PROMPT_COUNT prompts"
fi

# ─── Step: Pre-compute FARCA annotations ─────────────────────────
((STEP++))
log "Step $STEP/$TOTAL_STEPS: Pre-computing FARCA claim annotations..."
if [[ -f "data/oco_converted/train.jsonl" ]]; then
    FARCA_ARGS=(
        --input data/oco_converted/train.jsonl
        --output data/farca_annotations/claims.jsonl
    )
    if [[ -n "${PIPELINE_CONFIG:-}" ]]; then
        FARCA_ARGS+=(--config "$PIPELINE_CONFIG")
    fi

    uv run python -m src.phase4c_rl.scripts.precompute_farca_annotations \
        "${FARCA_ARGS[@]}" \
        >> "$RUN_LOG" 2>&1 || {
        log "  WARNING: FARCA annotation pre-computation failed (non-fatal)"
    }

    if [[ -f "data/farca_annotations/annotation_stats.json" ]]; then
        CLAIM_COUNT=$(python3 -c "import json; print(json.load(open('data/farca_annotations/annotation_stats.json')).get('total_claims', 0))" 2>/dev/null || echo "?")
        MEAN_REL=$(python3 -c "import json; print(json.load(open('data/farca_annotations/annotation_stats.json')).get('mean_reliability', 0))" 2>/dev/null || echo "?")
        log "  Claims extracted: $CLAIM_COUNT (mean reliability: $MEAN_REL)"
    fi
else
    log "  WARNING: data/oco_converted/train.jsonl not found, skipping FARCA precompute"
fi

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
log "=========================================="
log "Pipeline complete ($STEP/$TOTAL_STEPS steps)."
if $REFRESH_CORPUS; then
    log "  Phase A: Offsec corpus refresh completed"
fi
log "  Phase B: R2 sync + RL data prep completed"
if $CHANGED; then
    log "  Data was updated — downstream training can use fresh data."
else
    log "  No changes detected — data is already up to date."
fi
log "  Full log: $RUN_LOG"
log "=========================================="

# ─── Prune old logs (keep last 30) ───────────────────────────────
if [[ -d "$LOG_DIR" ]]; then
    ls -1t "$LOG_DIR"/run_*.log 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null || true
fi
