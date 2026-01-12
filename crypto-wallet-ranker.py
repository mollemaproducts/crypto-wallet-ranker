# ==========================================
# Wallet Scoring System Documentation
# ==========================================
#
# Scoring Components:
# -------------------
# - total_pnl        : Total profit and loss from all trades. Higher PnL increases the score.
# - sharpe           : Sharpe ratio (risk-adjusted return). Indicates consistency of returns.
#                     *Capped at a maximum value to prevent extreme outliers from dominating the score.*
# - buy_count        : Number of buy trades executed.
# - sell_count       : Number of sell trades executed.
# - num_realized     : Number of trades that have been closed / realized.
# - max_drawdown     : Maximum observed drawdown (risk measure). Higher drawdowns reduce the score.
# - win_rate         : Percentage of winning trades. Higher win rate increases the score.
# - tier             : Classification based on score, e.g., "A+", "A", "B".
# - score            : Overall composite score calculated from the above metrics.
#
# Smoothing and Normalization:
# -----------------------------
# 1. Metrics such as total_pnl, sharpe, and win_rate may be normalized or scaled to a uniform range
#    before combining into the composite score, to prevent any single metric from dominating.
# 2. Sharpe ratio is capped at a maximum value (e.g., 5) to avoid extreme values inflating the score.
# 3. Optional smoothing is applied to reduce volatility impact:
#    - Scores can be averaged over recent trades or weighted to avoid sudden jumps from single large trades.
#    - Max_drawdown may be weighted more heavily if high drawdowns are observed.
#
# Example of Score Calculation:
# -----------------------------
# score = (
#     normalized_total_pnl * weight_pnl +
#     capped_sharpe * weight_sharpe +
#     win_rate * weight_win_rate -
#     max_drawdown * weight_risk
# )
# Tier assignment:
# - A+ : score >= 100
# - A  : score 75–99
# - B  : score 50–74
# - ... additional tiers can be added as needed
#
# Note: The exact weights and normalization functions can be tuned to match desired sensitivity.

import json
import os
import time
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
from tabulate import tabulate  # Importing tabulate for pretty tables

# ---- CONFIG ----
DATA_DIR = "data/raw"            # partitioned JSON files directory
OUTPUT_FILE = "data/top_wallets.json"
LOOKBACK_DAYS = 180
RECENT_DAYS = 30
MIN_REALIZED_TRADES = 3
MAX_DRAWDOWN = 0.6
TOP_N = 25
SECONDS_IN_DAY = 86400
SHARPE_CAP = 5.0                 # cap to prevent extreme Sharpe scores
PNL_SMOOTH_DAYS = 30             # smooth over recent trades for PnL

# ---------- helpers ----------
def trade_ts(t):
    """Return timestamp in seconds."""
    if t.get("createdAt"):
        return datetime.fromisoformat(t["createdAt"].replace("Z", "+00:00")).timestamp()
    ts = t.get("timestamp")
    if ts:
        ts = float(ts)
        return ts / 1000 if ts > 1e12 else ts
    return 0

def load_partitioned_trades(days_back=LOOKBACK_DAYS):
    """Load trades from partitioned files within the lookback window."""
    cutoff = time.time() - days_back * SECONDS_IN_DAY
    trades = []

    # Traverse through each year folder
    for year_folder in sorted(os.listdir(DATA_DIR)):
        year_path = os.path.join(DATA_DIR, year_folder)
        if not os.path.isdir(year_path):
            continue

        # Traverse through each month folder within the year folder
        for month_folder in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month_folder)
            if not os.path.isdir(month_path):
                continue

            # Check if the year and month folder is within the lookback period
            try:
                folder_year_month = datetime.strptime(f"{year_folder}/{month_folder}", "%Y/%m")
            except ValueError:
                continue  # Skip folders that don't match the expected format

            if folder_year_month.timestamp() < cutoff:
                continue

            # Load trades from each file in the month folder
            for fname in sorted(os.listdir(month_path)):
                if not fname.endswith(".json"):
                    continue
                file_path = os.path.join(month_path, fname)
                with open(file_path, "r") as f:
                    bucket_trades = json.load(f)
                for t in bucket_trades:
                    if trade_ts(t) >= cutoff:
                        trades.append(t)

    print(f"Loaded {len(trades)} trades from the last {days_back} days")
    return trades

# ---------- grouping ----------
def group_wallets(trades, recent_days=RECENT_DAYS):
    now = time.time()
    recent_cutoff = now - recent_days * SECONDS_IN_DAY
    wallets = defaultdict(list)
    active_wallets = set()

    for t in trades:
        w = t.get("proxyWallet") or t.get("user")
        if not w:
            continue
        wallets[w].append(t)
        if trade_ts(t) >= recent_cutoff:
            active_wallets.add(w)

    return wallets, active_wallets

# ---------- stats ----------
def compute_stats(trades, smooth_days=PNL_SMOOTH_DAYS):
    positions = defaultdict(list)
    pnl = []
    buy = sell = 0
    now = time.time()
    smooth_cutoff = now - smooth_days * SECONDS_IN_DAY

    for t in sorted(trades, key=trade_ts):
        side = t.get("side", "").upper()
        size = float(t.get("size", 0))
        price = float(t.get("price", 0))
        key = f"{t.get('market_slug')}:{t.get('outcome')}"

        if side == "BUY":
            positions[key].append((size, price))
            buy += 1
        elif side == "SELL":
            sell += 1
            rem = size
            p = 0
            for bsize, bprice in positions[key]:
                if rem <= 0:
                    break
                take = min(rem, bsize)
                p += take * (price - bprice)
                rem -= take
            if p != 0:
                # Only consider recent trades for smoothing
                ts = trade_ts(t)
                if ts >= smooth_cutoff:
                    pnl.append(p)

    if not pnl:
        return None

    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    drawdown = np.max(peak - cum)
    wins = sum(1 for x in pnl if x > 0)

    total_pnl = sum(pnl)
    sharpe = np.mean(pnl) / (np.std(pnl) or 1)
    # cap extreme Sharpe
    sharpe = min(sharpe, SHARPE_CAP)

    return {
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "buy_count": buy,
        "sell_count": sell,
        "num_realized": len(pnl),
        "max_drawdown": drawdown,
        "win_rate": wins / len(pnl)
    }

# ---------- scoring ----------
def mm(x, lo, hi):
    return (x - lo) / (hi - lo) if hi > lo else 0

def compute_scores(candidates):
    pnls = [s["total_pnl"] for _, s in candidates]
    sharpes = [s["sharpe"] for _, s in candidates]
    dds = [s["max_drawdown"] for _, s in candidates]
    activities = [s["buy_count"] + s["sell_count"] for _, s in candidates]

    G = {
        "pnl_min": min(pnls),
        "pnl_max": max(pnls),
        "sh_min": min(sharpes),
        "sh_max": max(sharpes),
        "dd_max": max(dds),
        "activity_max": max(activities),
    }

    def score(s):
        v = 0
        v += 0.15 * mm(s["total_pnl"], G["pnl_min"], G["pnl_max"])       # PnL
        v += 0.40 * mm(s["sharpe"], G["sh_min"], G["sh_max"])             # Sharpe
        v += 0.20 * (1 - mm(s["max_drawdown"], 0, G["dd_max"]))           # Drawdown
        v += 0.10 * mm(s["buy_count"] + s["sell_count"], 1, G["activity_max"])  # Activity
        v += 0.15 * s["win_rate"]                                          # Win rate

        # limit trade count multiplier to prevent tiny wallets from inflating score
        multiplier = min(np.log1p(s["num_realized"]), 2)
        v *= multiplier
        return round(v * 100, 2)

    def tier(sc):
        if sc >= 80:
            return "A+"
        if sc >= 65:
            return "A"
        if sc >= 50:
            return "B"
        return "IGNORE"

    ranked = []
    for w, s in candidates:
        sc = score(s)
        t = tier(sc)
        if t != "IGNORE":
            ranked.append({
                "wallet": w,
                "score": sc,
                "tier": t,
                **{k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()}
            })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:TOP_N]

# ---------- main ----------
def main():
    trades = load_partitioned_trades()
    wallets, active_wallets = group_wallets(trades)

    candidates = []
    for w, ts in wallets.items():
        if w not in active_wallets:
            continue
        s = compute_stats(ts)
        if not s:
            continue
        if s["num_realized"] < MIN_REALIZED_TRADES or s["max_drawdown"] > MAX_DRAWDOWN:
            continue
        candidates.append((w, s))

    if not candidates:
        print("No wallets passed filters")
        return

    ranked = compute_scores(candidates)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(ranked, f, indent=2)

    # Pretty-print the ranked wallets to the console as a table, sorted by score descending
    headers = ["Wallet", "Score", "Tier", "Total PnL", "Sharpe", "Buy Count", "Sell Count", "Realized Trades", "Max Drawdown", "Win Rate"]
    table = [
        [w["wallet"], w["score"], w["tier"], w["total_pnl"], w["sharpe"], w["buy_count"], w["sell_count"], w["num_realized"], w["max_drawdown"], f"{w['win_rate']:.2f}"]
        for w in ranked
    ]
    print("Ranked Wallets (sorted by score descending):")
    print(tabulate(table, headers=headers, tablefmt="grid"))

    print(f"Saved {len(ranked)} wallets → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
