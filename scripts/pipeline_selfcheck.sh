#!/usr/bin/env bash
# Weekly pipeline selfcheck: re-runs key analysis steps on a sample to verify
# the pipeline is healthy on the current database. Commits updated CSVs if any
# changed and pushes to GitHub.
# Designed to be called by launchd every Sunday at 4am.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO/logs"
LOG="$LOG_DIR/selfcheck_$(date +%Y-%m-%d).log"
PY="$REPO/.venv/bin/python3"

mkdir -p "$LOG_DIR"

echo "=== pipeline_selfcheck.sh started: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
cd "$REPO"

run_step() {
    local label="$1"; shift
    echo "" | tee -a "$LOG"
    echo "--- $label ---" | tee -a "$LOG"
    if "$PY" "$@" 2>&1 | tee -a "$LOG"; then
        echo "[OK] $label" | tee -a "$LOG"
    else
        echo "[WARN] $label exited non-zero — check log" | tee -a "$LOG"
    fi
}

# Step 01: rebuild corpus parquet from updated DB (selfcheck = quick validation only)
run_step "01 corpus selfcheck" src/data/01_corpus_builder.py --selfcheck

# Step 06: verify split is still valid on a sample
run_step "06 split selfcheck" src/data/06_stratified_splitting.py --selfcheck --sample 50000

# Step 07: RF baseline on a sample (fast: 120k rows)
run_step "07 RF selfcheck" src/models/07_rf_baseline.py --selfcheck --sample 120000

# Step 08: comprehensive evaluation selfcheck
run_step "08 eval selfcheck" src/fairness/08_comprehensive_evaluation.py --selfcheck --sample 80000

# Step 10: reweighing selfcheck
run_step "10 reweigh selfcheck" src/fairness/10_preprocessing_mitigation.py --selfcheck --sample 80000

echo "" | tee -a "$LOG"
echo "=== Selfcheck steps complete ===" | tee -a "$LOG"

# Commit any updated selfcheck CSVs and push
cd "$REPO"
if git diff --quiet -- 'outputs/data/*_selfcheck.csv'; then
    echo "[INFO] No selfcheck CSVs changed — nothing to commit." | tee -a "$LOG"
else
    STAMP=$(date +%Y-%m-%d)
    git add outputs/data/*_selfcheck.csv 2>/dev/null || true
    git commit -m "Weekly selfcheck: updated pipeline outputs ${STAMP}" 2>&1 | tee -a "$LOG"
    git push origin main 2>&1 | tee -a "$LOG"
    echo "[INFO] Committed and pushed updated selfcheck CSVs." | tee -a "$LOG"
fi

echo "=== pipeline_selfcheck.sh finished: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

# Rotate logs older than 60 days
find "$LOG_DIR" -name "selfcheck_*.log" -mtime +60 -delete 2>/dev/null || true
