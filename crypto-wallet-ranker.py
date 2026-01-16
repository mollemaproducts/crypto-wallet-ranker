# ==========================================
# Wallet Scoring System with Graph Verification
# ==========================================

import json
import os
import shutil
import time
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
from tabulate import tabulate
import matplotlib.pyplot as plt

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
GRAPH_DIR = "data/graphs"
TOP_DATA_DIR = "data/top_wallet_data"
MAX_TRADES = 50  # limit to last 50 trades per wallet

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(TOP_DATA_DIR, exist_ok=True)

# Clear the graph directory
for f in os.listdir(GRAPH_DIR):
    path = os.path.join(GRAPH_DIR, f)
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

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
    """Load trades from partitioned files within the lookback window (YYYY/MM/DD)."""
    cutoff = time.time() - days_back * SECONDS_IN_DAY
    trades = []

    if not os.path.exists(DATA_DIR):
        return trades

    for year in sorted(os.listdir(DATA_DIR)):
        year_path = os.path.join(DATA_DIR, year)
        if not os.path.isdir(year_path): continue

        for month in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month)
            if not os.path.isdir(month_path): continue

            for day in sorted(os.listdir(month_path)):
                day_path = os.path.join(month_path, day)
                if not os.path.isdir(day_path): continue

                try:
                    folder_date = datetime.strptime(f"{year}/{month}/{day}", "%Y/%m/%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                if folder_date.timestamp() < cutoff: continue

                for fname in sorted(os.listdir(day_path)):
                    if not fname.endswith(".json"): continue
                    file_path = os.path.join(day_path, fname)
                    try:
                        with open(file_path, "r") as f:
                            bucket_trades = json.load(f)
                    except Exception:
                        continue

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
        if not w: continue
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
        side = t.get("side","").upper()
        size = float(t.get("size",0))
        price = float(t.get("price",0))
        key = f"{t.get('market_slug')}:{t.get('outcome')}"

        if side=="BUY":
            positions[key].append((size,price))
            buy += 1
        elif side=="SELL":
            sell += 1
            rem=size
            p=0
            for bsize,bprice in positions[key]:
                if rem<=0: break
                take=min(rem,bsize)
                p += take*(price-bprice)
                rem -= take
            if p != 0:
                ts = trade_ts(t)
                if ts >= smooth_cutoff:
                    pnl.append(p)

    if not pnl: return None

    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    drawdown = np.max(peak - cum)
    wins = sum(1 for x in pnl if x>0)

    total_pnl = sum(pnl)
    sharpe = np.mean(pnl)/(np.std(pnl) or 1)
    sharpe = min(sharpe, SHARPE_CAP)

    return {
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "buy_count": buy,
        "sell_count": sell,
        "num_realized": len(pnl),
        "max_drawdown": drawdown,
        "win_rate": wins/len(pnl)
    }

# ---------- scoring ----------
def mm(x, lo, hi):
    return (x-lo)/(hi-lo) if hi>lo else 0

def compute_scores(candidates):
    pnls = [s["total_pnl"] for _,s in candidates]
    sharpes = [s["sharpe"] for _,s in candidates]
    dds = [s["max_drawdown"] for _,s in candidates]
    activities = [s["buy_count"]+s["sell_count"] for _,s in candidates]

    G = {
        "pnl_min": min(pnls),
        "pnl_max": max(pnls),
        "sh_min": min(sharpes),
        "sh_max": max(sharpes),
        "dd_max": max(dds),
        "activity_max": max(activities)
    }

    def score(s):
        v = 0
        v += 0.15*mm(s["total_pnl"],G["pnl_min"],G["pnl_max"])
        v += 0.40*mm(s["sharpe"],G["sh_min"],G["sh_max"])
        v += 0.20*(1-mm(s["max_drawdown"],0,G["dd_max"]))
        v += 0.10*mm(s["buy_count"]+s["sell_count"],1,G["activity_max"])
        v += 0.15*s["win_rate"]
        multiplier = min(np.log1p(s["num_realized"]),2)
        v *= multiplier
        return round(v*100,2)

    def tier(sc):
        if sc>=80: return "A+"
        if sc>=65: return "A"
        if sc>=50: return "B"
        return "IGNORE"

    ranked=[]
    for w,s in candidates:
        sc=score(s)
        t=tier(sc)
        if t!="IGNORE":
            ranked.append({
                "wallet":w,
                "score":sc,
                "tier":t,
                **{k: round(v,2) if isinstance(v,float) else v for k,v in s.items()}
            })
    ranked.sort(key=lambda x:x["score"], reverse=True)
    return ranked[:TOP_N]

# ---------- verification & graphing ----------
def verify_wallet(wallet_address, trades, stats=None, index=None):
    wallet_trades = [t for t in trades if t.get("proxyWallet")==wallet_address or t.get("user")==wallet_address]
    if not wallet_trades:
        return

    positions = defaultdict(list)
    pnl = []
    timestamps = []

    for t in sorted(wallet_trades, key=trade_ts):
        side = t.get("side","").upper()
        size = float(t.get("size",0))
        price = float(t.get("price",0))
        key = f"{t.get('market_slug')}:{t.get('outcome')}"
        ts = trade_ts(t)

        if side=="BUY":
            positions[key].append((size,price))
        elif side=="SELL":
            rem=size
            p=0
            for bsize,bprice in positions[key]:
                if rem<=0: break
                take=min(rem,bsize)
                p += take*(price-bprice)
                rem -= take
            if p != 0:
                pnl.append(p)
                timestamps.append(datetime.fromtimestamp(ts))

    if not pnl: return

    cum_pnl = np.cumsum(pnl)
    # Save graph without showing
    plt.figure(figsize=(10,4))
    plt.plot(timestamps, cum_pnl, marker='o')
    plt.title(f"Wallet {wallet_address} - Cumulative PnL")
    plt.xlabel("Trade date")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"{index:02d}_{wallet_address}.png" if index is not None else f"{wallet_address}.png"
    plt.savefig(os.path.join(GRAPH_DIR, filename))
    plt.close()

    # --- Save wallet data + stats, minified and trimmed ---
    essential_trade_fields = ["market_slug","outcome","side","size","price","createdAt","timestamp"]

    trimmed_trades = [
        {k: t[k] for k in essential_trade_fields if k in t}
        for t in wallet_trades
    ]

    # Limit to last 50 trades
    trimmed_trades = trimmed_trades[-MAX_TRADES:]

    wallet_data = {
        "wallet": wallet_address,
        "stats": stats,
        "trades": trimmed_trades
    }

    data_filename = f"{index:02d}_{wallet_address}.json" if index is not None else f"{wallet_address}.json"
    with open(os.path.join(TOP_DATA_DIR, data_filename), "w") as f:
        json.dump(wallet_data, f, separators=(',', ':'))  # minified JSON

# ---------- main ----------
def main():
    trades = load_partitioned_trades()
    wallets, active_wallets = group_wallets(trades)

    candidates=[]
    for w,ts in wallets.items():
        if w not in active_wallets: continue
        s=compute_stats(ts)
        if not s: continue
        if s["num_realized"]<MIN_REALIZED_TRADES or s["max_drawdown"]>MAX_DRAWDOWN: continue
        candidates.append((w,s))

    if not candidates:
        print("No wallets passed filters")
        return

    ranked = compute_scores(candidates)

    # Save minified top wallets JSON
    with open(OUTPUT_FILE,"w") as f:
        json.dump(ranked, f, separators=(',', ':'))

    headers = ["#","Wallet","Score","Tier","Total PnL","Sharpe","Buy Count","Sell Count","Realized Trades","Max Drawdown","Win Rate"]
    table=[]
    for idx, w in enumerate(ranked,start=1):
        table.append([
            idx,
            w["wallet"],
            w["score"],
            w["tier"],
            w["total_pnl"],
            w["sharpe"],
            w["buy_count"],
            w["sell_count"],
            w["num_realized"],
            w["max_drawdown"],
            f"{w['win_rate']:.2f}"
        ])
    print("Ranked Wallets (sorted by score descending):")
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print(f"Saved {len(ranked)} wallets → {OUTPUT_FILE}")

    # Generate verification graphs & save top wallet data
    for idx, w in enumerate(ranked[:TOP_N], start=1):
        verify_wallet(w["wallet"], trades, stats=w, index=idx)

if __name__=="__main__":
    main()
