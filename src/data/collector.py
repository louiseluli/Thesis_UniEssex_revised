import os
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow direct script run

from typing import List, Tuple, Optional
import argparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.database import get_conn, DB_FILE  # upgraded DB utils (WAL/FKs/ctxmgr)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
API_BASE_URL = "https://api.redtube.com/"
API_DAILY_LIMIT = 29999          # safety margin below the hard 30k/day
REQUEST_DELAY = 0.35             # polite pacing between requests (seconds)

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # align with database.py
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "collector_state.json"

TIMEOUT = (10, 30)  # (connect, read) seconds

# Tune stop rules: be more patient on genuinely new/low-count categories
LOW_COUNT_LIMIT = 100
NEW_CAT_DUP_LIMIT = 9999
EXISTING_CAT_DUP_LIMIT = 9999

# --- NEW: API result slicing (see RedTube docs: 'ordering' + optional 'period') ---
# Allowed: ordering ∈ {'newest','mostviewed','rating'}
# Allowed: period   ∈ {'weekly','monthly','alltime'}  (only when ordering is set)
ORDERING = None     # e.g., 'newest'
PERIOD   = None     # e.g., 'monthly' (only used when ORDERING is set)



# -----------------------------------------------------------------------------
# Robust session with retries (idempotent GETs)
# -----------------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "msc-fairness-collector/1.0"})
    return s

# -----------------------------------------------------------------------------
# JSON state (fine-grained progress); DB 'collection_state' (authoritative rate)
# -----------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "categories": None,
        "current_category_index": 0,  # retained for backwards-compat
        "current_category": None,     # NEW: prefer resuming by category name
        "current_page": 1,
    }

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

# -----------------------------------------------------------------------------
# Rate limiting stored in DB: collection_state(day, requests_used, last_page_fetched, reset_at)
# -----------------------------------------------------------------------------
def _utc_today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _next_midnight_utc(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).date()
    return datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)

def load_or_init_rate_state(conn) -> dict:
    day = _utc_today_str()
    cur = conn.execute(
        "SELECT day, requests_used, last_page_fetched, reset_at FROM collection_state WHERE day = ?",
        (day,),
    )
    row = cur.fetchone()
    if row:
        return dict(row)
    reset_at = _next_midnight_utc().isoformat()
    conn.execute(
        "INSERT INTO collection_state(day, requests_used, last_page_fetched, reset_at) VALUES(?, ?, ?, ?)",
        (day, 0, 0, reset_at),
    )
    return {"day": day, "requests_used": 0, "last_page_fetched": 0, "reset_at": reset_at}

def can_consume_requests(conn, n: int = 1) -> Tuple[bool, int]:
    """
    Returns (allowed, sleep_seconds).
    If not allowed, you should sleep for sleep_seconds (until reset_at).
    """
    state = load_or_init_rate_state(conn)
    if state["requests_used"] + n <= API_DAILY_LIMIT:
        return True, 0
    reset_at = datetime.fromisoformat(state["reset_at"])
    now = datetime.now(timezone.utc)
    sleep_secs = max(1, int((reset_at - now).total_seconds()))
    return False, sleep_secs

def consume_requests(conn, n: int = 1) -> None:
    day = _utc_today_str()
    conn.execute(
        "UPDATE collection_state SET requests_used = requests_used + ? WHERE day = ?",
        (n, day),
    )

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def parse_duration_to_seconds(d: Optional[str]) -> Optional[int]:
    if not d:
        return None
    parts = [p for p in str(d).split(":") if p.strip().isdigit()]
    if not parts:
        print(f"[WARN] Could not parse duration: {d!r} — stored as NULL")
        return None
    parts = list(map(int, parts))
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = 0, 0, parts[0]
    return int(h) * 3600 + int(m) * 60 + int(s)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def get_existing_ids_for_category(conn, category: str) -> set:
    """
    Return a set of video_ids already stored for this category.
    Uses the video_categories table to keep it tight.
    """
    rows = conn.execute(
        "SELECT vc.video_id FROM video_categories vc WHERE vc.category = ?",
        (category,)
    ).fetchall()
    return {row["video_id"] for row in rows}



# Optional probing/estimation helpers (not used in main loop)
def probe_category_page(session: requests.Session, conn, category: str, page: int) -> int:
    try:
        videos, end = fetch_videos_for_category(session, conn, category, page)
        if end:
            return 0
        return len(videos or [])
    except RateLimitHit:
        raise
    except Exception:
        return 0

def find_last_page_index(session: requests.Session, conn, category: str) -> int:
    if probe_category_page(session, conn, category, 1) == 0:
        return 0
    lo, hi = 1, 2
    while probe_category_page(session, conn, category, hi) > 0:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if probe_category_page(session, conn, category, mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo

def estimate_category_count(session: requests.Session, conn, category: str) -> int:
    last_page = find_last_page_index(session, conn, category)
    if last_page == 0:
        return 0
    n_last = probe_category_page(session, conn, category, last_page)
    return (last_page - 1) * 20 + n_last

# -----------------------------------------------------------------------------
# API Calls
# -----------------------------------------------------------------------------
class RateLimitHit(Exception):
    """Raised when the API daily cap is hit (code 1005) or local cap logic denies more calls."""

def get_categories(session: requests.Session, conn) -> List[str]:
    print("Fetching category list from API...")
    allowed, sleep_secs = can_consume_requests(conn, n=1)
    if not allowed:
        raise RateLimitHit(f"Cap reached. Sleep {sleep_secs}s until reset.")

    params = {"data": "redtube.Categories.getCategoriesList", "output": "json"}
    resp = session.get(API_BASE_URL, params=params, timeout=TIMEOUT)
    consume_requests(conn, 1)

    if resp.status_code >= 400:
        raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()

    if isinstance(data, dict) and "code" in data:
        code = data.get("code")
        if code == 1005:
            raise RateLimitHit("API daily limit reached (1005).")
        raise RuntimeError(f"API error fetching categories: {data}")

    categories = [item["category"] for item in data.get("categories", []) if "category" in item]
    print(f"Found {len(categories)} categories.")
    return categories

def fetch_videos_for_category(session: requests.Session, conn, category: str, page: int):
    allowed, sleep_secs = can_consume_requests(conn, n=1)
    if not allowed:
        raise RateLimitHit(f"Cap reached. Sleep {sleep_secs}s until reset.")

    params = {
        "data": "redtube.Videos.searchVideos",
        "output": "json",
        "category": category,
        "page": page,
    }
    # Add ordering/period if set (per API docs)
    # ordering ∈ {newest, mostviewed, rating}; period ∈ {weekly, monthly, alltime} (only when ordering is used)
    if ORDERING:
        params["ordering"] = ORDERING
        if PERIOD:
            params["period"] = PERIOD

    resp = session.get(API_BASE_URL, params=params, timeout=TIMEOUT)
    consume_requests(conn, 1)


    if resp.status_code >= 400:
        raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()

    if isinstance(data, dict) and "code" in data:
        code = data.get("code")
        if code == 2001:
            return [], True
        if code == 1005:
            raise RateLimitHit("API daily limit reached (1005).")
        msg = data.get("message", "Unknown API error")
        raise RuntimeError(f"API error for '{category}' page {page}: {code} {msg}")

    videos = data.get("videos", [])
    if not videos:
        print(f"[info] Empty result set for '{category}' page {page} - likely end of content")
        return [], True

    return videos, False

# -----------------------------------------------------------------------------
# DB Writes (normalized: videos + video_tags + video_categories)
# -----------------------------------------------------------------------------
def save_videos_to_db(conn, videos: list, category_ctx: str) -> int:
    if not videos:
        return 0

    cur = conn.cursor()
    actually_new = 0  # only count genuinely NEW rows
    now = now_iso()

    for vwrap in videos:
        v = vwrap.get("video", {})
        vid = v.get("video_id")

        # Guard: skip any record without a usable primary key rather than
        # attempting to insert NULL and failing inside executemany.
        if not vid:
            print(f"[WARN] Skipping video with missing video_id in category '{category_ctx}'")
            continue

        existing = cur.execute("SELECT 1 FROM videos WHERE video_id = ?", (vid,)).fetchone()

        title = v.get("title")
        url = v.get("url")
        duration = parse_duration_to_seconds(v.get("duration"))
        views = v.get("views")
        rating = v.get("rating")
        ratings = v.get("ratings")
        publish_date = v.get("publish_date") or v.get("publishDate")
        is_active = 1

        cur.execute(
            """
            INSERT INTO videos(
                video_id, title, url, duration, views, rating, ratings,
                publish_date, category_source, is_active, retrieved_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title=excluded.title,
                url=excluded.url,
                duration=COALESCE(excluded.duration, videos.duration),
                views=COALESCE(excluded.views, videos.views),
                rating=COALESCE(excluded.rating, videos.rating),
                ratings=COALESCE(excluded.ratings, videos.ratings),
                publish_date=COALESCE(excluded.publish_date, videos.publish_date),
                category_source=excluded.category_source,
                is_active=excluded.is_active,
                retrieved_at=excluded.retrieved_at
            """,
            (vid, title, url, duration, views, rating, ratings,
             publish_date, category_ctx, is_active, now)
        )

        if not existing:
            actually_new += 1

        tags = []
        for t in (v.get("tags") or []):
            tag = t.get("tag_name") or t.get("tag")
            if tag:
                tags.append((vid, tag))
        if tags:
            cur.executemany(
                "INSERT OR IGNORE INTO video_tags(video_id, tag) VALUES(?, ?)", tags
            )

        cats = []
        for c in (v.get("categories") or []):
            cname = c.get("category") or c.get("category_name")
            if cname:
                cats.append((vid, cname))
        cats.append((vid, category_ctx))
        cur.executemany(
            "INSERT OR IGNORE INTO video_categories(video_id, category) VALUES(?, ?)", cats
        )

    conn.commit()
    return actually_new

def ensure_category_status_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS category_status(
            category    TEXT PRIMARY KEY,
            end_reached INTEGER NOT NULL DEFAULT 0,
            last_page   INTEGER NOT NULL DEFAULT 0,
            last_checked TEXT
        )
    """)
    conn.commit()

def upsert_category_progress(conn, category: str, last_page: int, end_reached: int = 0) -> None:
    now = now_iso()
    conn.execute("""
        INSERT INTO category_status(category, end_reached, last_page, last_checked)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(category) DO UPDATE SET
            end_reached=excluded.end_reached,
            last_page=excluded.last_page,
            last_checked=excluded.last_checked
    """, (category, end_reached, last_page, now))
    conn.commit()

def get_start_page_for_category(conn, category: str) -> int:
    ensure_category_status_table(conn)
    row = conn.execute(
        "SELECT last_page, end_reached FROM category_status WHERE category = ?",
        (category,)
    ).fetchone()
    if row and row["end_reached"] == 0 and int(row["last_page"]) > 0:
        return int(row["last_page"]) + 1
    return 1

def get_incomplete_categories(conn, categories: List[str]) -> List[str]:
    ensure_category_status_table(conn)
    incomplete = []
    cur = conn.cursor()
    for cat in categories:
        row = cur.execute(
            "SELECT end_reached, last_page FROM category_status WHERE category = ?",
            (cat,)
        ).fetchone()

        if row is None:
            incomplete.append(cat)
            print(f"[scan] Category '{cat}': no status - WILL FETCH")
        else:
            if row["end_reached"] == 0:
                incomplete.append(cat)
                lp = row["last_page"]
                print(f"[scan] Category '{cat}': not finished (last_page={lp}) - WILL FETCH")
            else:
                print(f"[scan] Category '{cat}': finished before - SKIPPING")
    return incomplete

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", help="Focus on specific category")
    parser.add_argument("--reset", action="store_true", help="Reset state and start fresh")
    parser.add_argument("--reopen", action="store_true", help="Reopen the focused category (end_reached=0,last_page=0)")
    parser.add_argument("--start-page", type=int, help="Override starting page (e.g., 60)")
    # keep your existing --ordering/--period adds if you already added them
    parser.add_argument("--ordering", choices=["newest", "mostviewed", "rating"], help="API ordering for result list")
    parser.add_argument("--period", choices=["weekly", "monthly", "alltime"], help="API period (valid only when --ordering is used)")
    parser.add_argument("--new-only", action="store_true",
                    help="Only persist videos not already in DB (skip updates)")
    args = parser.parse_args()

    print(f"[i] Using DB at: {DB_FILE}")
    session = make_session()

    if args.reset:
        STATE_FILE.unlink(missing_ok=True)
        print("[reset] Cleared state file")

    # >>> THIS MUST COME BEFORE ANY use of `state` <<<
    state = load_state()

    # >>> PUT THE OVERRIDES *AFTER* load_state AND *BEFORE* the DB context <<<
    global ORDERING, PERIOD
    if args.ordering:
        ORDERING = args.ordering
    if args.period:
        PERIOD = args.period

    with get_conn() as conn:
        # 1) Categories (resume-friendly)
        if state.get("categories") is None:
            try:
                cats = get_categories(session, conn)
                state["categories"] = cats
                state["current_category_index"] = 0
                state["current_category"] = None
                state["current_page"] = 1
                save_state(state)
            except RateLimitHit as e:
                rate = load_or_init_rate_state(conn)
                reset_at = datetime.fromisoformat(rate["reset_at"])
                now = datetime.now(timezone.utc)
                sleep_secs = max(1, int((reset_at - now).total_seconds()))
                print(f"[cap] {e}. Sleeping {sleep_secs}s until {reset_at} (UTC).")
                time.sleep(sleep_secs)
                return
            except Exception as ex:
                print(f"[error] Cannot fetch categories: {ex}")
                return

        # 2) Optionally focus on one category
        cats = state["categories"]
        if args.category:
            if args.category in cats:
                cats = [args.category]
                print(f"[focus] Targeting only category: {args.category}")
                # NEW: --reopen resets DB status for the focused category
                if args.reopen:
                    conn.execute(
                        "UPDATE category_status SET end_reached=0, last_page=0 WHERE category = ?",
                        (args.category,)
                    )
                    conn.commit()
                    # also clear JSON resume pointer so page=1 is respected
                    STATE_FILE.unlink(missing_ok=True)
                    print(f"[reopen] Reset status for '{args.category}' (end_reached=0, last_page=0)")
            else:
                print(f"[error] Category '{args.category}' not found!")
                return


        # 3) Filter to incomplete categories only
        incomplete_cats = get_incomplete_categories(conn, cats)
        if not incomplete_cats:
            print("[done] All categories appear complete!")
            return

        # 4) Decide starting index based on current_category name (not index)
        start_idx = 0
        if state.get("current_category") in incomplete_cats:
            start_idx = incomplete_cats.index(state["current_category"])

        for i in range(start_idx, len(incomplete_cats)):
            cat = incomplete_cats[i]
            state["current_category"] = cat
            save_state(state)

            # Show how many we already have for this category
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as cnt FROM video_categories WHERE category = ?", (cat,))
            existing_count = cur.fetchone()["cnt"]
            print(f"\n--- Category: {cat} ({i+1}/{len(incomplete_cats)}) - Already have {existing_count} videos ---")
            
            # NEW: choose duplicate-page stop threshold per category
            dup_limit = NEW_CAT_DUP_LIMIT if existing_count < LOW_COUNT_LIMIT else EXISTING_CAT_DUP_LIMIT
            print(f"[info] Duplicate-page stop limit for '{cat}': {dup_limit} (existing_count={existing_count})")
            # Pre-load existing IDs for fast "new-only" filtering
            existing_ids = get_existing_ids_for_category(conn, cat)




            # Decide starting page
            page = max(
                get_start_page_for_category(conn, cat),
                state.get("current_page", 1) if state.get("current_category") == cat else 1
            )
            # NEW: explicit override from CLI (keeps everything else intact)
            if args.start_page and args.start_page > 0:
                page = args.start_page

            # Ensure status row exists and reflect that we are about to attempt `page`
            upsert_category_progress(conn, cat, last_page=page-1, end_reached=0)


            videos_this_session = 0
            empty_page_streak = 0
            duplicate_page_streak = 0

            while True:
                print(f"[req] Fetching page {page} for '{cat}' (session total: {videos_this_session})...")
                try:
                    videos, end_of_pages = fetch_videos_for_category(session, conn, cat, page)
                except RateLimitHit as e:
                    # Persist exact resume point and exit cleanly
                    rate = load_or_init_rate_state(conn)
                    reset_at = datetime.fromisoformat(rate["reset_at"])
                    now = datetime.now(timezone.utc)
                    sleep_secs = max(1, int((reset_at - now).total_seconds()))
                    upsert_category_progress(conn, cat, last_page=page-1, end_reached=0)
                    state["current_page"] = page
                    state["current_category"] = cat
                    save_state(state)
                    print(f"[cap] {e}. Sleeping {sleep_secs}s until {reset_at} (UTC).")
                    time.sleep(sleep_secs)
                    return
                except (requests.RequestException, ValueError) as netex:
                    print(f"[net] Network/parse error on '{cat}' page {page}: {netex}. Retrying in 60s.")
                    time.sleep(60)
                    continue
                except Exception as ex:
                    print(f"[error] Unexpected error: {ex}. Retrying in 60s.")
                    time.sleep(60)
                    continue

                if end_of_pages:
                    print(f"[done] No more videos for '{cat}'. Total this session: {videos_this_session}")
                    upsert_category_progress(conn, cat, last_page=page, end_reached=1)
                    # prepare for next category
                    state["current_page"] = 1
                    state["current_category"] = None
                    save_state(state)
                    break

                # New-only pre-filter (skip already-known IDs if the flag is on)
                videos_to_save = videos
                if args.new_only:
                    videos_to_save = [
                        vw for vw in videos
                        if vw.get("video", {}).get("video_id") not in existing_ids
                    ]

                inserted = save_videos_to_db(conn, videos_to_save, cat)
                upsert_category_progress(conn, cat, last_page=page, end_reached=0)

                # Keep the in-memory set fresh to avoid re-inserting within this run
                if videos_to_save:
                    for vw in videos_to_save:
                        vid = vw.get("video", {}).get("video_id")
                        if vid:
                            existing_ids.add(vid)

                # Log based on what we actually tried to save
                print(f"[db] Found {len(videos)} videos, attempted to save {len(videos_to_save)}; "
                      f"{inserted} were NEW, {len(videos_to_save)-inserted} were updates (skipped when --new-only).")

                if len(videos) == 0:
                    empty_page_streak += 1
                    duplicate_page_streak = 0
                    print(f"[warn] Page {page} was empty. Empty streak: {empty_page_streak}")
                    if empty_page_streak >= 3:
                        print(f"[done] 3 consecutive empty pages for '{cat}' - assuming end")
                        upsert_category_progress(conn, cat, last_page=page, end_reached=1)
                        state["current_page"] = 1
                        state["current_category"] = None
                        save_state(state)
                        break
                else:
                    empty_page_streak = 0
                    if inserted == 0:
                        duplicate_page_streak += 1
                        print(f"[warn] Page {page} had 0 new videos (all duplicates). Duplicate streak: {duplicate_page_streak}")
                        if duplicate_page_streak >= dup_limit:
                            print(f"[done] {duplicate_page_streak} consecutive pages with only duplicates for '{cat}' (limit={dup_limit}) - marking as complete")
                            upsert_category_progress(conn, cat, last_page=page, end_reached=1)
                            state["current_page"] = 1
                            state["current_category"] = None
                            save_state(state)
                            break
                    else:
                        duplicate_page_streak = 0
                        videos_this_session += inserted


                # advance: persist next page in JSON
                page += 1
                state["current_page"] = page
                state["current_category"] = cat
                save_state(state)
                time.sleep(REQUEST_DELAY)

    print("\n--- Collection complete for available categories/pages at this run. ---")

if __name__ == "__main__":
    main()
