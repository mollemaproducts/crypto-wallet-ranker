import websocket
import json
import time
import requests

# ------------------------
# 1️⃣ Configuration
# ------------------------
WS_URL = "wss://polygon-mainnet.g.alchemy.com/v2/DO046I-D8ytSDqKgz4wuI"  # Replace with your Alchemy WebSocket URL
RPC_URL = "https://polygon-mainnet.g.alchemy.com/v2/DO046I-D8ytSDqKgz4wuI"  # HTTP RPC for fetching full block
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
TOP_WALLETS = [w.lower() for w in TOP_WALLETS]  # normalize addresses

# ------------------------
# 2️⃣ WebSocket Callbacks
# ------------------------
def on_message(ws, message):
    """
    Called when a new block header is received
    """
    data = json.loads(message)

    if "method" in data and data["method"] == "eth_subscription":
        block_hash = data["params"]["result"]["hash"]
        block_number = int(data["params"]["result"]["number"], 16)
        print(f"New Block: {block_number} - {block_hash}")

        # Fetch full block transactions
        try:
            resp = requests.post(RPC_URL, json={
                "jsonrpc": "2.0",
                "method": "eth_getBlockByHash",
                "params": [block_hash, True],
                "id": 1
            })
            result = resp.json().get("result")
            if not result:
                print(f"Block {block_hash} not available yet, skipping...")
                return

            txs = result.get("transactions", [])
            for tx in txs:
                tx_from = tx.get("from", "").lower()
                tx_to = tx.get("to", "").lower()
                if tx_from in TOP_WALLETS or tx_to in TOP_WALLETS:
                    print(f">>> Wallet event detected: {tx_from} → {tx_to}")
                    print("Tx Hash:", tx.get("hash"))

        except Exception as e:
            print("Failed to fetch block transactions:", e)

def on_error(ws, error):
    print("WebSocket Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Closed:", close_status_code, close_msg)

def on_open(ws):
    """
    Subscribe to new blocks
    """
    print("WebSocket connection opened.")
    subscribe_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_subscribe",
        "params": ["newHeads"]
    }
    ws.send(json.dumps(subscribe_msg))
    print("Subscribed to new blocks.")

# ------------------------
# 3️⃣ Start WebSocket
# ------------------------
if __name__ == "__main__":
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
            print("WebSocket connection failed, retrying in 5 seconds...", e)
            time.sleep(5)
