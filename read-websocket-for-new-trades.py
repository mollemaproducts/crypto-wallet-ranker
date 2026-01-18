import websocket
import json
import time
import requests
import os
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import shutil

# ------------------------
# 1️⃣ Configuration
# ------------------------
WS_URL = "wss://polygon-mainnet.g.alchemy.com/v2/DO046I-D8ytSDqKgz4wuI"
RPC_URL = "https://polygon-mainnet.g.alchemy.com/v2/DO046I-D8ytSDqKgz4wuI"
TOP_WALLETS = [
    "0xd44e29936409019f93993de8bd603ef6cb1bb15e",
    "0x86a29f88fcc23ea2b0e01b4e186b043e26c873a8",
    "0x76b2672c87681acd8e311f58f334f5e592886999",
    "0xce0a9a3c325820a078fda1f74fc973b7c16cc836",
    "0xf26574811c82461066318a7880440226447f99c9",
    "0xba59425a6f9146cca291e0d40da0ecaa7bd7a430",
    "0x3df8ba1356edba28e08762e026ebfaa0cc92369f",
    "0x4d59e4c467ae134978e5955e70195006be9d9a12",
    "0x4ffe49ba2a4cae123536a8af4fda48faeb609f71",
    "0x674b949283ea2eb33c2cad519a3e2f26b3144f07",
    "0xc184a1eadc93ab3b569af9373465f13d8af4a5a2",
    "0x58c03b6be218494943a48f08b3bd8718a1d0887b",
    "0xeb04552a914c126f8569644b906c3e035b9bbc3c",
    "0x0781872fe78deb6ad8147e482952b4a5461ec4ff",
    "0x9b19e731ae2d00a4635f46841a3b301d77f26b1c"
]
TOP_WALLETS = [w.lower() for w in TOP_WALLETS]  # normalize

DATA_DIR = "data/live_trading/live_trades"
GRAPH_DIR = "data/live_trading/graphs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# Clean graphs folder before generating new ones
shutil.rmtree(GRAPH_DIR)
os.makedirs(GRAPH_DIR, exist_ok=True)

# Store trades per wallet
wallet_trades = {w: [] for w in TOP_WALLETS}
wallet_index = {w: i+1 for i, w in enumerate(TOP_WALLETS)}  # index starting at 1

# ------------------------
# 2️⃣ Helpers
# ------------------------
def save_wallet_trades(wallet):
    """Save all trades for a wallet in the live_trading folder"""
    w_data = wallet_trades[wallet]
    if not w_data:
        return
    filename = os.path.join(DATA_DIR, f"{wallet_index[wallet]}_{wallet}.json")
    with open(filename, "w") as f:
        json.dump(w_data, f, indent=2)

def plot_wallet_pnl(wallet):
    """Generate a PnL graph for the wallet silently"""
    trades = wallet_trades[wallet]
    if not trades:
        return
    pnl = []
    cum = 0
    for t in trades:
        side = t.get("side", "").upper()
        size = float(t.get("size", 0))
        price = float(t.get("price", 0))
        change = size * price if side == "BUY" else -size * price
        cum += change
        pnl.append(cum)
    plt.figure(figsize=(6, 3))
    plt.plot(pnl, label=wallet)
    plt.xlabel("Trade #")
    plt.ylabel("Cumulative PnL")
    plt.title(f"Wallet {wallet} Performance")
    plt.tight_layout()
    graph_file = os.path.join(GRAPH_DIR, f"{wallet_index[wallet]}_{wallet}.png")
    plt.savefig(graph_file)
    plt.close()

def process_tx(tx):
    tx_from = (tx.get("from") or "").lower()
    tx_to = (tx.get("to") or "").lower()
    involved = None
    if tx_from in TOP_WALLETS:
        involved = tx_from
    elif tx_to in TOP_WALLETS:
        involved = tx_to
    if not involved:
        return
    # Save trade
    wallet_trades[involved].append(tx)
    save_wallet_trades(involved)
    plot_wallet_pnl(involved)

# ------------------------
# 3️⃣ WebSocket Callbacks
# ------------------------
def on_message(ws, message):
    data = json.loads(message)
    if "method" in data and data["method"] == "eth_subscription":
        block_hash = data["params"]["result"]["hash"]
        block_number = int(data["params"]["result"]["number"], 16)
        print(f"[WS] New block {block_number}")
        fetch_block_transactions(block_hash)

def on_error(ws, error):
    print("[WS] Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("[WS] Closed:", close_status_code, close_msg)

def on_open(ws):
    print("[WS] Connection opened")
    subscribe_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_subscribe",
        "params": ["newHeads"]
    }
    ws.send(json.dumps(subscribe_msg))
    print("[WS] Subscribed to new blocks")

# ------------------------
# 4️⃣ Block fetching
# ------------------------
def fetch_block_transactions(block_hash):
    try:
        resp = requests.post(RPC_URL, json={
            "jsonrpc": "2.0",
            "method": "eth_getBlockByHash",
            "params": [block_hash, True],
            "id": 1
        })
        result = resp.json().get("result")
        if not result:
            return
        for tx in result.get("transactions", []):
            process_tx(tx)
    except Exception as e:
        print("[RPC] Failed:", e)

# ------------------------
# 5️⃣ Polling fallback (every 3s)
# ------------------------
def polling_loop():
    last_block = None
    while True:
        try:
            resp = requests.post(RPC_URL, json={
                "jsonrpc": "2.0",
                "method": "eth_blockNumber",
                "params": [],
                "id": 1
            })
            block_number = int(resp.json()["result"], 16)
            if last_block is None or block_number > last_block:
                for b in range(last_block+1 if last_block else block_number, block_number+1):
                    resp_block = requests.post(RPC_URL, json={
                        "jsonrpc": "2.0",
                        "method": "eth_getBlockByNumber",
                        "params": [hex(b), True],
                        "id": 1
                    })
                    result = resp_block.json().get("result")
                    if result:
                        for tx in result.get("transactions", []):
                            process_tx(tx)
                last_block = block_number
        except Exception as e:
            print("[Polling] Error:", e)
        time.sleep(3)

# ------------------------
# 6️⃣ Start WebSocket + Polling
# ------------------------
if __name__ == "__main__":
    import threading
    # Start polling fallback in a separate thread
    threading.Thread(target=polling_loop, daemon=True).start()
    while True:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever()
        except Exception as e:
            print("[WS] Connection failed, retrying...", e)
            time.sleep(5)
