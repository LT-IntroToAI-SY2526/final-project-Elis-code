import json
import glob
import time


# =========================================================
# LOAD ABIDES ORDER BOOK LOGS
# =========================================================

def load_abides_orderbook(path):
    """
    Loads a single ABIDES order book JSON file.
    Expected format:
      {
        "timestamp": 123456789,
        "bids": [[price, size], ...],
        "asks": [[price, size], ...]
      }
    """
    with open(path, "r") as f:
        return json.load(f)


# =========================================================
# CONVERT ABIDES LOGS → SNAPSHOTS.JSONL
# =========================================================

def convert_abides_to_snapshots(
    abides_folder="abides_logs/",
    output_path="snapshots.jsonl",
    markets=("EX1", "EX2", "EX3")
):
    """
    Converts ABIDES order book logs into the same format used by trainer.py.

    ABIDES folder structure example:
      abides_logs/EX1/orderbook_*.json
      abides_logs/EX2/orderbook_*.json
      abides_logs/EX3/orderbook_*.json

    Output format (one JSON per line):
      {
        "t": timestamp,
        "binance": {...},
        "coinbase": {...},
        "kraken": {...}
      }
    """

    # Map ABIDES markets → your project’s exchange names
    exchange_map = {
        markets[0]: "binance",
        markets[1]: "coinbase",
        markets[2]: "kraken"
    }

    # Gather all orderbook files for each market
    market_files = {
        m: sorted(glob.glob(f"{abides_folder}/{m}/orderbook_*.json"))
        for m in markets
    }

    # Ensure all markets have the same number of snapshots
    num_snapshots = min(len(files) for files in market_files.values())

    print(f"Found {num_snapshots} synchronized snapshots across markets.")

    with open(output_path, "w") as out:
        for i in range(num_snapshots):
            snapshot = {"t": time.time()}

            for abides_market, files in market_files.items():
                ob = load_abides_orderbook(files[i])
                project_name = exchange_map[abides_market]

                snapshot[project_name] = {
                    "bids": ob.get("bids", []),
                    "asks": ob.get("asks", [])
                }

            out.write(json.dumps(snapshot) + "\n")

            if (i + 1) % 50 == 0 or i == 0:
                print(f"Converted snapshot {i+1}/{num_snapshots}")

    print(f"✅ Conversion complete → {output_path}")


if __name__ == "__main__":
    convert_abides_to_snapshots(
        abides_folder="abides_logs",
        output_path="snapshots.jsonl",
        markets=("EX1", "EX2", "EX3")
    )
