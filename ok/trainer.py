import json
import math
import random
import numpy as np
from typing import List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# LOAD SNAPSHOTS
# =========================================================

def load_snapshots(path, max_lines=None):
    data = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            data.append(json.loads(line))
    return data


# =========================================================
# ORDER BOOK UTILITIES
# =========================================================

def simulate_market_order(order_book, side, quantity):
    bids = order_book["bids"]
    asks = order_book["asks"]

    if side == "buy":
        levels = asks
    else:
        levels = bids

    remaining = quantity
    cost = 0.0
    filled = 0.0

    for price, size in levels:
        trade_size = min(size, remaining)
        cost += trade_size * price
        filled += trade_size
        remaining -= trade_size
        if remaining <= 0:
            break

    if filled < quantity:
        return None, None

    vwap = cost / filled
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = 0.5 * (best_bid + best_ask)

    if side == "buy":
        slippage = (vwap - mid) / mid
    else:
        slippage = (mid - vwap) / mid

    return vwap, slippage


def label_liquidity(order_book, side="buy", quantity=0.01, max_slippage=0.002):
    vwap, slippage = simulate_market_order(order_book, side, quantity)
    if vwap is None:
        return 0
    return 1 if slippage <= max_slippage else 0


def features_from_order_book(order_book, depth_levels=5):
    bids = order_book["bids"][:depth_levels]
    asks = order_book["asks"][:depth_levels]

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    spread = best_ask - best_bid
    mid = 0.5 * (best_bid + best_ask)

    bid_sizes = [s for _, s in bids]
    ask_sizes = [s for _, s in asks]

    return np.array([
        spread / mid,
        sum(bid_sizes),
        sum(ask_sizes),
        max(bid_sizes),
        max(ask_sizes),
        best_bid,
        best_ask
    ], dtype=float)


# =========================================================
# LIQUIDITY MODEL
# =========================================================

def build_liquidity_dataset(order_books):
    X, y = [], []
    for ob in order_books:
        X.append(features_from_order_book(ob))
        y.append(label_liquidity(ob))
    return np.vstack(X), np.array(y)


def train_liquidity_model(order_books):
    X, y = build_liquidity_dataset(order_books)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Liquidity model accuracy:", model.score(X_test, y_test))
    return model


# =========================================================
# ARBITRAGE PREDICTION MODEL
# =========================================================

def current_arbitrage_edge(ob1, ob2):
    b1 = ob1["bids"][0][0]
    a1 = ob1["asks"][0][0]
    b2 = ob2["bids"][0][0]
    a2 = ob2["asks"][0][0]
    return max(b1 - a2, b2 - a1)


def build_arb_dataset(snapshots, horizon=5, threshold=0.5):
    X, y = [], []
    for i in range(len(snapshots) - horizon):
        cur = snapshots[i]
        future = snapshots[i+1:i+1+horizon]

        f_bin = features_from_order_book(cur["binance"])
        f_cb = features_from_order_book(cur["coinbase"])
        f_kr = features_from_order_book(cur["kraken"])

        feat = np.concatenate([f_bin, f_cb, f_kr])

        future_edges = []
        for f in future:
            e1 = current_arbitrage_edge(f["binance"], f["coinbase"])
            e2 = current_arbitrage_edge(f["binance"], f["kraken"])
            e3 = current_arbitrage_edge(f["coinbase"], f["kraken"])
            future_edges.append(max(e1, e2, e3))

        label = 1 if max(future_edges) >= threshold else 0

        X.append(feat)
        y.append(label)

    return np.vstack(X), np.array(y)


def train_arb_model(snapshots):
    X, y = build_arb_dataset(snapshots)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    print("Arbitrage prediction accuracy:", model.score(X_test, y_test))
    return model


# =========================================================
# DEMO PIPELINE
# =========================================================

def demo(path="snapshots.jsonl"):
    snapshots = load_snapshots(path, max_lines=2000)

    print("\nTraining liquidity model (Binance)...")
    liq_model = train_liquidity_model([s["binance"] for s in snapshots])

    print("\nTraining arbitrage prediction model...")
    arb_model = train_arb_model(snapshots)

    sample = random.choice(snapshots)
    feat = np.concatenate([
        features_from_order_book(sample["binance"]),
        features_from_order_book(sample["coinbase"]),
        features_from_order_book(sample["kraken"])
    ]).reshape(1, -1)

    prob = arb_model.predict_proba(feat)[0][1]
    pred = arb_model.predict(feat)[0]

    print("\nDemo:")
    print("Arbitrage predicted:", bool(pred), "prob:", prob)


if __name__ == "__main__":
    demo("snapshots.jsonl")
