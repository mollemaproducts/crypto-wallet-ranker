# ==========================================
# Wallet Scoring & Copy-Trading Eligibility
# Split Skill Leaderboard vs Copy-Safe List
# ==========================================

import json, os, shutil, time
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
import matplotlib.pyplot as plt

class WalletRanker:
    def __init__(self, data_dir="data/raw", skill_file="data/skill_wallets.json", copy_file="data/copy_safe_wallets.json"):
        self.DATA_DIR = data_dir
        self.SKILL_OUTPUT_FILE = skill_file
        self.COPY_OUTPUT_FILE = copy_file

        # timing & limits
        self.LOOKBACK_DAYS = 180
        self.RECENT_DAYS = 30
        self.SECONDS_IN_DAY = 86400
        self.MIN_REALIZED_TRADES = 1
        self.MAX_DRAWDOWN = 0.6
        self.TOP_N = 25
        self.SHARPE_CAP = 5.0
        self.MAX_TRADES = 50

        # folders
        self.GRAPH_DIR = "data/graphs"
        self.TOP_DATA_DIR = "data/top_wallet_data"
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.GRAPH_DIR, exist_ok=True)
        os.makedirs(self.TOP_DATA_DIR, exist_ok=True)

        # clear old graphs
        for f in os.listdir(self.GRAPH_DIR):
            p = os.path.join(self.GRAPH_DIR, f)
            if os.path.isfile(p): os.remove(p)
            else: shutil.rmtree(p)

    # ---------- helpers ----------
    def trade_ts(self, t):
        if t.get("createdAt"):
            return datetime.fromisoformat(t["createdAt"].replace("Z","+00:00")).timestamp()
        ts = float(t.get("timestamp",0))
        return ts/1000 if ts > 1e12 else ts

    # ---------- loader ----------
    def load_partitioned_trades(self, days_back=None):
        if days_back is None:
            days_back = self.LOOKBACK_DAYS
        cutoff = time.time() - days_back * self.SECONDS_IN_DAY
        trades = []
        for y in os.listdir(self.DATA_DIR):
            for m in os.listdir(os.path.join(self.DATA_DIR,y)):
                for d in os.listdir(os.path.join(self.DATA_DIR,y,m)):
                    try:
                        folder_ts = datetime.strptime(f"{y}/{m}/{d}", "%Y/%m/%d").replace(tzinfo=timezone.utc).timestamp()
                    except:
                        continue
                    if folder_ts < cutoff: continue
                    path = os.path.join(self.DATA_DIR,y,m,d)
                    for fn in os.listdir(path):
                        if not fn.endswith(".json"): continue
                        with open(os.path.join(path,fn)) as f:
                            try: bucket = json.load(f)
                            except: continue
                            for t in bucket:
                                if self.trade_ts(t) >= cutoff:
                                    trades.append(t)
        print(f"Loaded {len(trades)} trades")
        return trades

    # ---------- grouping ----------
    def group_wallets(self, trades):
        now = time.time()
        recent_cutoff = now - self.RECENT_DAYS * self.SECONDS_IN_DAY
        wallets = defaultdict(list)
        active = set()
        for t in trades:
            w = t.get("proxyWallet") or t.get("user")
            if not w: continue
            wallets[w].append(t)
            if self.trade_ts(t) >= recent_cutoff:
                active.add(w)
        return wallets, active

    # ---------- stats ----------
    def compute_stats(self, trades):
        positions = defaultdict(list)
        pnl_events = []
        buy = sell = 0

        for t in sorted(trades, key=self.trade_ts):
            side = t.get("side","").upper()
            size = float(t.get("size",0))
            price = float(t.get("price",0))
            ts = self.trade_ts(t)
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
                    if b[0] <= 0: positions[key].pop(0)
                if realized != 0: pnl_events.append((ts, realized))

        if len(pnl_events) < self.MIN_REALIZED_TRADES: return None

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

        sharpe = (np.mean(daily_pnl)/(np.std(daily_pnl) or 1))*np.sqrt(365)
        sharpe = min(sharpe, self.SHARPE_CAP)

        return {
            "total_pnl": float(np.sum(daily_pnl)),
            "sharpe": float(sharpe),
            "buy_count": buy,
            "sell_count": sell,
            "num_realized": len(daily_pnl),
            "active_days": len(daily),
            "max_drawdown": drawdown,
            "win_rate": float(np.mean(daily_pnl>0))
        }

    # ---------- copy-trading metrics ----------
    def timing_entropy(self, trades):
        ts = sorted(self.trade_ts(t) for t in trades)
        if len(ts)<10: return 0.0
        deltas = np.diff(ts)
        hist,_ = np.histogram(deltas,bins=20)
        p = hist/(hist.sum() or 1)
        return float(-np.sum(p*np.log(p+1e-9)))

    def median_hold_time(self, trades):
        positions=defaultdict(list)
        holds=[]
        for t in sorted(trades, key=self.trade_ts):
            key=(t.get("market_slug"), t.get("outcome"))
            side=t.get("side","").upper()
            size=float(t.get("size",0))
            ts=self.trade_ts(t)
            if side=="BUY":
                positions[key].append([ts,size])
            elif side=="SELL":
                rem=size
                for pos in positions.get(key,[]):
                    if rem<=0: break
                    take=min(rem,pos[1])
                    holds.append(ts-pos[0])
                    pos[1]-=take
                    rem-=take
        return float(np.median(holds)) if holds else 0.0

    def size_cv(self, trades):
        sizes=[float(t.get("size",0)) for t in trades if float(t.get("size",0))>0]
        if len(sizes)<5: return 0.0
        return float(np.std(sizes)/np.mean(sizes))

    def manual_trade_likelihood(self, trades):
        ent = self.timing_entropy(trades)
        cv = self.size_cv(trades)
        score = ent * cv
        return min(score/2.0,1.0)

    def copy_trading_score(self, trades, stats):
        ent = self.timing_entropy(trades)
        hold = self.median_hold_time(trades)
        cv = self.size_cv(trades)
        manual = self.manual_trade_likelihood(trades)
        score=1.0
        if ent<1.0: score*=0.5
        if cv<0.05: score*=0.6
        if hold<30: score*=0.7
        if stats["active_days"]<10: score*=0.7
        if stats["win_rate"]<0.45: score*=0.8
        score *= (1.0 - 0.5*manual)
        verdict="ELIGIBLE" if score>=0.6 else "RISKY" if score>=0.3 else "DO_NOT_COPY"
        return {
            "copy_score":round(score,3),
            "copy_verdict":verdict,
            "timing_entropy":round(ent,2),
            "median_hold_sec":int(hold),
            "size_cv":round(cv,3),
            "manual_score":round(manual,3)
        }

    # ---------- scoring ----------
    def compute_scores(self, candidates, performance_only=False):
        if not candidates: return []

        def mm(x, a, b): return (x - a)/(b - a) if b>a else 0

        pnls = [s["total_pnl"] for _, s in candidates]
        shs  = [s["sharpe"] for _, s in candidates]
        dds  = [s["max_drawdown"] for _, s in candidates]
        days = [s["active_days"] for _, s in candidates]

        G = {
            "pnl_min": min(pnls),
            "pnl_max": max(pnls),
            "sh_min": min(shs),
            "sh_max": max(shs),
            "dd_max": max(dds),
            "days_max": max(days)
        }

        def score(s):
            if performance_only:
                # Skill leaderboard only cares about performance
                v = 0
                v += 0.25 * mm(s["sharpe"], G["sh_min"], G["sh_max"])
                v += 0.20 * mm(s["total_pnl"], G["pnl_min"], G["pnl_max"])
                v += 0.20 * (1 - mm(s["max_drawdown"], 0, G["dd_max"]))
                v += 0.20 * mm(s["active_days"], 1, G["days_max"])
                v += 0.15 * s["win_rate"]
                return round(v * min(np.log1p(s["active_days"]),2)*100,2)
            else:
                # Copy-safe score uses copy_trading_score + activity
                return round(s.get("copy_score",0)*100,2)

        def tier(sc):
            if sc>=110: return "A+++"
            if sc>=95: return "A++"
            if sc>=80: return "A+"
            if sc>=65: return "A"
            if sc>=45: return "B"
            if sc>=25: return "C"
            return "IGNORE"

        ranked=[]
        for w,s in candidates:
            sc = score(s)
            t = tier(sc) if performance_only else ("COPY" if s.get("copy_verdict")=="ELIGIBLE" else "RISKY")
            if t!="IGNORE":
                ranked.append({"wallet":w,"score":sc,"tier":t,**{k:round(v,2) if isinstance(v,float) else v for k,v in s.items()}})
        ranked.sort(key=lambda x:x["score"],reverse=True)
        return ranked[:self.TOP_N]

    # ---------- verification & saving ----------
    def verify_wallet(self, wallet,trades,stats,index):
        wallet_trades=[t for t in trades if t.get("proxyWallet")==wallet or t.get("user")==wallet]
        if not wallet_trades: return
        positions=defaultdict(list)
        pnl=[]
        ts_list=[]
        for t in sorted(wallet_trades,key=self.trade_ts):
            side=t.get("side","").upper()
            size=float(t.get("size",0))
            price=float(t.get("price",0))
            key=f"{t.get('market_slug')}:{t.get('outcome')}"
            if side=="BUY": positions[key].append((size,price))
            elif side=="SELL":
                rem=size; p=0
                for bsize,bprice in positions[key]:
                    if rem<=0: break
                    take=min(rem,bsize)
                    p+=take*(price-bprice)
                    rem-=take
                if p!=0:
                    pnl.append(p)
                    ts_list.append(datetime.fromtimestamp(self.trade_ts(t)))
        if pnl:
            plt.figure(figsize=(10,4))
            plt.plot(ts_list,np.cumsum(pnl),marker="o")
            plt.title(f"{wallet} – Cumulative PnL")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.GRAPH_DIR,f"{index:02d}_{wallet}.png"))
            plt.close()
        trimmed=wallet_trades[-self.MAX_TRADES:]
        out={"wallet":wallet,"stats":stats,"trades":trimmed}
        with open(os.path.join(self.TOP_DATA_DIR,f"{index:02d}_{wallet}.json"),"w") as f:
            json.dump(out,f,separators=(',',':'))

    # ---------- run ----------
    def run(self):
        trades = self.load_partitioned_trades()
        wallets, active = self.group_wallets(trades)

        eligible_candidates=[]
        risky_candidates=[]
        for w,ts in wallets.items():
            if w not in active: continue
            s=self.compute_stats(ts)
            if not s: continue
            if s["max_drawdown"]>self.MAX_DRAWDOWN: continue
            copy=self.copy_trading_score(ts,s)
            wallet_data={**s,**copy}
            if copy["copy_verdict"]=="DO_NOT_COPY": continue
            elif copy["copy_verdict"]=="ELIGIBLE": eligible_candidates.append((w,wallet_data))
            elif copy["copy_verdict"]=="RISKY": risky_candidates.append((w,wallet_data))

        # --- Skill Leaderboard (performance only) ---
        skill_ranked = self.compute_scores(eligible_candidates+risky_candidates, performance_only=True)
        with open(self.SKILL_OUTPUT_FILE,"w") as f: json.dump(skill_ranked,f,separators=(',',':'))
        print(f"Saved Skill Leaderboard ({len(skill_ranked)}) → {self.SKILL_OUTPUT_FILE}")

        # --- Copy-Safe List (eligible only) ---
        copy_ranked = self.compute_scores(eligible_candidates, performance_only=False)
        with open(self.COPY_OUTPUT_FILE,"w") as f: json.dump(copy_ranked,f,separators=(',',':'))
        print(f"Saved Copy-Safe List ({len(copy_ranked)}) → {self.COPY_OUTPUT_FILE}")

        # --- verify & save per-wallet ---
        for i,w in enumerate(skill_ranked+copy_ranked,1):
            self.verify_wallet(w["wallet"],trades,w,i)

        print(f"Saved per-wallet trade files → {self.TOP_DATA_DIR}")


# ---------- MAIN ----------
if __name__=="__main__":
    ranker = WalletRanker(
        data_dir="data/raw",
        skill_file="data/skill_wallets.json",
        copy_file="data/copy_safe_wallets.json"
    )
    ranker.run()
