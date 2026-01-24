# ==========================================
# Wallet Scoring System (Fixed + Verified)
# ==========================================

import json, os, shutil, time
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
from tabulate import tabulate
import matplotlib.pyplot as plt

# ---- CONFIG ----
DATA_DIR = "data/raw"
OUTPUT_FILE = "data/top_wallets.json"

LOOKBACK_DAYS = 180
RECENT_DAYS = 30
SECONDS_IN_DAY = 86400

MIN_REALIZED_TRADES = 1
MAX_DRAWDOWN = 0.6
TOP_N = 25

SHARPE_CAP = 5.0
MAX_TRADES = 50

GRAPH_DIR = "data/wallet-ranker/graphs"
TOP_DATA_DIR = "data/wallet-ranker/top_wallet_data"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(TOP_DATA_DIR, exist_ok=True)

# clear graphs
for f in os.listdir(GRAPH_DIR):
    p = os.path.join(GRAPH_DIR, f)
    if os.path.isfile(p): os.remove(p)
    else: shutil.rmtree(p)

# ---------- helpers ----------
def trade_ts(t):
    if t.get("createdAt"):
        return datetime.fromisoformat(
            t["createdAt"].replace("Z","+00:00")
        ).timestamp()
    ts = float(t.get("timestamp",0))
    return ts/1000 if ts > 1e12 else ts

# ---------- loader ----------
def load_partitioned_trades(days_back=LOOKBACK_DAYS):
    cutoff = time.time() - days_back * SECONDS_IN_DAY
    trades = []

    for y in os.listdir(DATA_DIR):
        for m in os.listdir(os.path.join(DATA_DIR,y)):
            for d in os.listdir(os.path.join(DATA_DIR,y,m)):
                try:
                    folder_ts = datetime.strptime(
                        f"{y}/{m}/{d}", "%Y/%m/%d"
                    ).replace(tzinfo=timezone.utc).timestamp()
                except:
                    continue
                if folder_ts < cutoff: continue

                path = os.path.join(DATA_DIR,y,m,d)
                for fn in os.listdir(path):
                    if not fn.endswith(".json"): continue
                    with open(os.path.join(path,fn)) as f:
                        try:
                            bucket = json.load(f)
                        except:
                            continue
                        for t in bucket:
                            if trade_ts(t) >= cutoff:
                                trades.append(t)

    print(f"Loaded {len(trades)} trades")
    return trades

# ---------- grouping ----------
def group_wallets(trades):
    now = time.time()
    recent_cutoff = now - RECENT_DAYS * SECONDS_IN_DAY
    wallets = defaultdict(list)
    active = set()

    for t in trades:
        w = t.get("proxyWallet") or t.get("user")
        if not w: continue
        wallets[w].append(t)
        if trade_ts(t) >= recent_cutoff:
            active.add(w)

    return wallets, active

# ---------- stats ----------
def compute_stats(trades):
    positions = defaultdict(list)
    pnl_events = []
    buy = sell = 0

    for t in sorted(trades, key=trade_ts):
        side = t.get("side","").upper()
        size = float(t.get("size",0))
        price = float(t.get("price",0))
        ts = trade_ts(t)
        key = f"{t.get('market_slug')}:{t.get('outcome')}"

        if side == "BUY":
            positions[key].append([size, price])
            buy += 1

        elif side == "SELL":
            sell += 1
            rem = size
            realized = 0

            while rem > 0 and positions[key]:
                b = positions[key][0]
                take = min(rem, b[0])
                realized += take * (price - b[1])
                b[0] -= take
                rem -= take
                if b[0] <= 0:
                    positions[key].pop(0)

            if realized != 0:
                pnl_events.append((ts, realized))

    if len(pnl_events) < MIN_REALIZED_TRADES:
        return None

    daily = defaultdict(float)
    for ts,p in pnl_events:
        d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        daily[d] += p

    daily_pnl = np.array(list(daily.values()))
    cap = np.percentile(np.abs(daily_pnl), 95)
    daily_pnl = np.clip(daily_pnl, -cap, cap)

    cum = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(cum)
    drawdown = float(np.max(peak - cum))

    sharpe = (np.mean(daily_pnl) / (np.std(daily_pnl) or 1)) * np.sqrt(365)
    sharpe = min(sharpe, SHARPE_CAP)

    return {
        "total_pnl": float(np.sum(daily_pnl)),
        "sharpe": float(sharpe),
        "buy_count": buy,
        "sell_count": sell,
        "num_realized": len(daily_pnl),
        "active_days": len(daily),
        "max_drawdown": drawdown,
        "win_rate": float(np.mean(daily_pnl > 0))
    }

# ---------- scoring ----------
def compute_scores(candidates):
    def mm(x,a,b): return (x-a)/(b-a) if b>a else 0

    pnls = [s["total_pnl"] for _,s in candidates]
    shs  = [s["sharpe"] for _,s in candidates]
    dds  = [s["max_drawdown"] for _,s in candidates]
    days = [s["active_days"] for _,s in candidates]

    G = {
        "pnl_min": min(pnls), "pnl_max": max(pnls),
        "sh_min": min(shs),   "sh_max": max(shs),
        "dd_max": max(dds),
        "days_max": max(days)
    }

    def score(s):
        v = 0
        v += 0.25 * mm(s["sharpe"], G["sh_min"], G["sh_max"])
        v += 0.20 * mm(s["total_pnl"], G["pnl_min"], G["pnl_max"])
        v += 0.20 * (1 - mm(s["max_drawdown"], 0, G["dd_max"]))
        v += 0.20 * mm(s["active_days"], 1, G["days_max"])
        v += 0.15 * s["win_rate"]
        return round(v * min(np.log1p(s["active_days"]),2) * 100, 2)

    def tier(sc):
        if sc >= 110: return "A+++"
        if sc >= 95:  return "A++"
        if sc >= 80:  return "A+"
        if sc >= 65:  return "A"
        if sc >= 45:  return "B"
        if sc >= 25:  return "C"
        return "IGNORE"

    ranked=[]
    for w,s in candidates:
        sc = score(s)
        t  = tier(sc)
        if t!="IGNORE":
            ranked.append({
                "wallet":w,"score":sc,"tier":t,
                **{k:round(v,2) if isinstance(v,float) else v for k,v in s.items()}
            })

    ranked.sort(key=lambda x:x["score"], reverse=True)
    return ranked[:TOP_N]

# ---------- verification & saving ----------
def verify_wallet(wallet, trades, stats, index):
    wallet_trades = [
        t for t in trades
        if t.get("proxyWallet")==wallet or t.get("user")==wallet
    ]

    if not wallet_trades:
        return

    # cumulative PnL graph
    positions = defaultdict(list)
    pnl = []
    ts_list = []

    for t in sorted(wallet_trades, key=trade_ts):
        side = t.get("side","").upper()
        size = float(t.get("size",0))
        price = float(t.get("price",0))
        key = f"{t.get('market_slug')}:{t.get('outcome')}"

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
            if p!=0:
                pnl.append(p)
                ts_list.append(datetime.fromtimestamp(trade_ts(t)))

    if pnl:
        plt.figure(figsize=(10,4))
        plt.plot(ts_list, np.cumsum(pnl), marker="o")
        plt.title(f"{wallet} – Cumulative PnL")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIR, f"{index:02d}_{wallet}.png"))
        plt.close()

    trimmed = wallet_trades[-MAX_TRADES:]
    out = {
        "wallet": wallet,
        "stats": stats,
        "trades": trimmed
    }

    with open(os.path.join(TOP_DATA_DIR, f"{index:02d}_{wallet}.json"), "w") as f:
        json.dump(out, f, separators=(',',':'))

# ---------- main ----------
def main():
    trades = load_partitioned_trades()
    wallets, active = group_wallets(trades)

    candidates=[]
    for w,ts in wallets.items():
        if w not in active: continue
        s = compute_stats(ts)
        if not s: continue
        if s["max_drawdown"] > MAX_DRAWDOWN: continue
        candidates.append((w,s))

    if not candidates:
        print("No wallets passed filters")
        return

    ranked = compute_scores(candidates)

    with open(OUTPUT_FILE,"w") as f:
        json.dump(ranked,f,separators=(',',':'))

    print(f"Saved {len(ranked)} wallets → {OUTPUT_FILE}")

    for i,w in enumerate(ranked,1):
        verify_wallet(w["wallet"], trades, w, i)

    print(f"Saved per-wallet trade files → {TOP_DATA_DIR}")

if __name__=="__main__":
    main()
