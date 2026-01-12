import requests
import json
import os
import time
from datetime import datetime, timezone
import hashlib

# ---- CONFIG ----
DATA_API = "https://data-api.polymarket.com/trades"
LIMIT = 500
BASE_DIR = "data/raw"  # base directory for partitioned files
INDEX_FILE = "data/index/seen_ids.json"
PARTITION_MINUTES = 30  # partition size
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs("data/index", exist_ok=True)

# ---- HELPERS ----
def trade_timestamp_seconds(t):
    """Get trade timestamp in seconds."""
    if t.get("createdAt"):
        return datetime.fromisoformat(t["createdAt"].replace("Z", "+00:00")).timestamp()
    ts = t.get("timestamp")
    if ts:
        ts = float(ts)
        return ts / 1000 if ts > 1e12 else ts
    return 0

def trade_id(t):
    """Compute a unique ID for a trade if not provided."""
    if t.get("id"):
        return str(t["id"])
    raw = f"{trade_timestamp_seconds(t)}-{t.get('user')}-{t.get('market_slug')}-{t.get('side')}-{t.get('size')}"
    return hashlib.sha1(raw.encode()).hexdigest()

def current_bucket():
    """Get current time bucket rounded down to PARTITION_MINUTES."""
    now = datetime.now(timezone.utc)
    minute = (now.minute // PARTITION_MINUTES) * PARTITION_MINUTES
    return now.replace(minute=minute, second=0, microsecond=0)

def current_month_folder():
    """Get the folder path for the current year and month."""
    now = datetime.now(timezone.utc)
    year_month = now.strftime("%Y/%m")
    path = os.path.join(BASE_DIR, year_month)
    os.makedirs(path, exist_ok=True)
    return path

def load_seen():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_seen(seen):
    with open(INDEX_FILE, "w") as f:
        json.dump(list(seen), f)

# ---- COLLECTOR ----
def fetch_trades():
    r = requests.get(DATA_API, params={"limit": LIMIT})
    r.raise_for_status()
    return r.json()

def save_partitioned_trades(trades):
    """Save trades into partitioned JSON files, avoiding duplicates and organizing by month/year."""
    seen = load_seen()
    bucket = current_bucket().strftime("%Y-%m-%d_%H-%M")
    month_folder = current_month_folder()
    path = os.path.join(month_folder, f"trades_{bucket}.json")

    new_trades = []
    for t in trades:
        tid = trade_id(t)
        if tid in seen:
            continue
        seen.add(tid)
        t["_id"] = tid
        new_trades.append(t)

    if not new_trades:
        print("No new trades to store")
        return

    # Load existing bucket if exists
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    combined = existing + new_trades
    combined.sort(key=trade_timestamp_seconds)

    with open(path, "w") as f:
        json.dump(combined, f, indent=2)

    save_seen(seen)

    print(
        f"Stored {len(new_trades)} new trades in {path} | "
        f"Total in this bucket: {len(combined)} | "
        f"Last trade: {datetime.fromtimestamp(trade_timestamp_seconds(combined[-1]), tz=timezone.utc)}"
    )

# ---- MAIN ----
def main():
    trades = fetch_trades()
    if not trades:
        print("No trades fetched")
        return
    save_partitioned_trades(trades)

def run_collector_loop():
    while True:
        try:
            main()  # your collector function
        except Exception as e:
            print("Error during collection:", e)
        time.sleep(60)  # wait 60 seconds before next fetch

if __name__ == "__main__":
    run_collector_loop()
