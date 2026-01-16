import requests
import json
import os
import time
import hashlib
from datetime import datetime, timezone


class PolymarketTradeCollector:
    # ---- CONFIG ----
    DATA_API = "https://data-api.polymarket.com/trades"
    LIMIT = 500
    BASE_DIR = "data/raw"
    INDEX_FILE = "data/index/seen_ids.json"
    PARTITION_MINUTES = 30
    SLEEP_SECONDS = 60

    def __init__(self):
        os.makedirs(self.BASE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.INDEX_FILE), exist_ok=True)

    # ---- HELPERS ----
    @staticmethod
    def trade_timestamp_seconds(t):
        """Get trade timestamp in seconds."""
        if t.get("createdAt"):
            return datetime.fromisoformat(
                t["createdAt"].replace("Z", "+00:00")
            ).timestamp()

        ts = t.get("timestamp")
        if ts:
            ts = float(ts)
            return ts / 1000 if ts > 1e12 else ts

        return 0

    @staticmethod
    def trade_id(t):
        """Compute a deterministic unique ID for a trade."""
        if t.get("id"):
            return str(t["id"])

        raw = (
            f"{PolymarketTradeCollector.trade_timestamp_seconds(t)}-"
            f"{t.get('user')}-"
            f"{t.get('market_slug')}-"
            f"{t.get('side')}-"
            f"{t.get('size')}"
        )
        return hashlib.sha1(raw.encode()).hexdigest()

    def current_bucket(self):
        """Round current time down to PARTITION_MINUTES."""
        now = datetime.now(timezone.utc)
        minute = (now.minute // self.PARTITION_MINUTES) * self.PARTITION_MINUTES
        return now.replace(minute=minute, second=0, microsecond=0)

    def current_day_folder(self):
        """Return data/raw/YYYY/MM/DD and ensure it exists."""
        now = datetime.now(timezone.utc)
        path = os.path.join(
            self.BASE_DIR,
            now.strftime("%Y"),
            now.strftime("%m"),
            now.strftime("%d"),
        )
        os.makedirs(path, exist_ok=True)
        return path

    # ---- SEEN IDS ----
    def load_seen(self):
        if os.path.exists(self.INDEX_FILE):
            with open(self.INDEX_FILE, "r") as f:
                return set(json.load(f))
        return set()

    def save_seen(self, seen):
        with open(self.INDEX_FILE, "w") as f:
            json.dump(list(seen), f)

    # ---- DATA FETCH ----
    def fetch_trades(self):
        r = requests.get(self.DATA_API, params={"limit": self.LIMIT}, timeout=15)
        r.raise_for_status()
        return r.json()

    # ---- STORAGE ----
    def save_partitioned_trades(self, trades):
        seen = self.load_seen()
        bucket = self.current_bucket().strftime("%Y-%m-%d_%H-%M")
        day_folder = self.current_day_folder()
        path = os.path.join(day_folder, f"trades_{bucket}.json")

        new_trades = []
        for t in trades:
            tid = self.trade_id(t)
            if tid in seen:
                continue

            seen.add(tid)
            t["_id"] = tid
            new_trades.append(t)

        if not new_trades:
            print("No new trades to store")
            return

        if os.path.exists(path):
            with open(path, "r") as f:
                existing = json.load(f)
        else:
            existing = []

        combined = existing + new_trades
        combined.sort(key=self.trade_timestamp_seconds)

        with open(path, "w") as f:
            json.dump(combined, f, indent=2)

        self.save_seen(seen)

        last_ts = self.trade_timestamp_seconds(combined[-1])
        print(
            f"Stored {len(new_trades)} new trades in {path} | "
            f"Total: {len(combined)} | "
            f"Last trade: {datetime.fromtimestamp(last_ts, tz=timezone.utc)}"
        )

    # ---- RUNNERS ----
    def run_once(self):
        trades = self.fetch_trades()
        if not trades:
            print("No trades fetched")
            return
        self.save_partitioned_trades(trades)

    def run_forever(self):
        while True:
            try:
                self.run_once()
            except Exception as e:
                print("Collector error:", e)
            time.sleep(self.SLEEP_SECONDS)


# ---- ENTRY POINT ----
if __name__ == "__main__":
    collector = PolymarketTradeCollector()
    collector.run_forever()
