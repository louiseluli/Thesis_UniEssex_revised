#!/usr/bin/env bash
# Daily incremental collection: picks up new videos from the API (newest-first).
# Rate limit: 29,999 requests/day (enforced by collector.py via DB state).
# Designed to be called by launchd at 3am daily.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO/logs"
LOG="$LOG_DIR/collect_$(date +%Y-%m-%d).log"
PY="$REPO/.venv/bin/python3"

mkdir -p "$LOG_DIR"

echo "=== collect_daily.sh started: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
echo "Working directory: $REPO" | tee -a "$LOG"

cd "$REPO"

# Verify DB exists
if [ ! -f "data/redtube_videos.db" ]; then
    echo "[FATAL] Database not found at data/redtube_videos.db — aborting." | tee -a "$LOG"
    exit 1
fi

# Print current row count before collection
BEFORE=$("$PY" -c "import sqlite3; c=sqlite3.connect('data/redtube_videos.db'); print(c.execute('SELECT COUNT(*) FROM videos').fetchone()[0])")
echo "[INFO] Videos before collection: $BEFORE" | tee -a "$LOG"

# Run collector: newest ordering, skip already-known IDs
"$PY" src/data/collector.py \
    --ordering newest \
    --new-only \
    2>&1 | tee -a "$LOG"

# Print updated row count
AFTER=$("$PY" -c "import sqlite3; c=sqlite3.connect('data/redtube_videos.db'); print(c.execute('SELECT COUNT(*) FROM videos').fetchone()[0])")
NEW=$(( AFTER - BEFORE ))
echo "[INFO] Videos after collection: $AFTER (added: $NEW)" | tee -a "$LOG"
echo "=== collect_daily.sh finished: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

# Rotate logs older than 30 days
find "$LOG_DIR" -name "collect_*.log" -mtime +30 -delete 2>/dev/null || true
