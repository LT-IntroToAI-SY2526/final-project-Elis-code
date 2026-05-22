import math
import random
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# 1. SIMPLE TRIANGULAR ARBITRAGE DETECTION
# ---------------------------------------------------------

def triangular_arbitrage_profit(rate_ab, rate_bc, rate_ca):
    """
    rate_ab: A -> B
    rate_bc: B -> C
    rate_ca: C -> A
    returns: profit factor ( >1 means arbitrage )
    """
    start_amount = 1.0
    after_b = start_amount * rate_ab
    after_c = after_b * rate_bc
    final_a = after_c * rate_ca
    return final_a


def has_triangular_arbitrage(rate_ab, rate_bc, rate_ca, threshold=1.0001):
    """
    Returns True if arbitrage exists.
    threshold avoids floating‑point noise.
    """
    profit = triangular_arbitrage_profit(rate_ab, rate_bc, rate_ca)
    return profit > threshold, profit


# ---------------------------------------------------------
# 2. GRAPH-BASED ARBITRAGE (BELLMAN–FORD NEGATIVE CYCLE)
# ---------------------------------------------------------

def build_graph_from_rates(rates):
    """
    rates: dict of dicts
    Example:
    {
        'A': {'B': rate_ab, 'C': rate_ac},
        'B': {'A': rate_ba, 'C': rate_bc},
        'C': {'A': rate_ca, 'B': rate_cb}
    }
    """
    edges = []
    for u in rates:
        for v in rates[u]:
            rate = rates[u][v]
            weight = -math.log(rate)
            edges.append((u, v, weight))
    return edges


def bellman_ford_negative_cycle(nodes, edges, source):
    """
    Returns True if a negative cycle exists.
    """
    dist = {node: float('inf') for node in nodes}
    dist[source] = 0.0

    # Relax edges |V|-1 times
    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    # Check for negative cycle
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return True

    return False


def has_graph_arbitrage(rates):
    nodes = list(rates.keys())
    edges = build_graph_from_rates(rates)
    source = nodes[0]
    return bellman_ford_negative_cycle(nodes, edges, source)


# ---------------------------------------------------------
# 3. REAL FX DATA FETCHING
# ---------------------------------------------------------

def get_fx_rates(base="USD", symbols=("EUR", "JPY")):
    """
    Fetch real FX rates for base -> symbols.
    Uses exchangerate.host (free, no key required).
    """
    url = f"https://api.exchangerate.host/latest?base={base}&symbols={','.join(symbols)}"
    resp = requests.get(url)
    data = resp.json()
    return data["rates"]


def build_triangular_from_real_fx():
    """
    Builds A,B,C = USD, EUR, JPY using real FX data.
    Returns rate_ab, rate_bc, rate_ca.
    """
    # Get USD->EUR and USD->JPY
    usd_rates = get_fx_rates(base="USD", symbols=("EUR", "JPY"))
    usd_to_eur = usd_rates["EUR"]
    usd_to_jpy = usd_rates["JPY"]

    # Get EUR->USD and EUR->JPY
    eur_rates = get_fx_rates(base="EUR", symbols=("USD", "JPY"))
    eur_to_usd = eur_rates["USD"]
    eur_to_jpy = eur_rates["JPY"]

    # Get JPY->USD and JPY->EUR
    jpy_rates = get_fx_rates(base="JPY", symbols=("USD", "EUR"))
    jpy_to_usd = jpy_rates["USD"]
    jpy_to_eur = jpy_rates["EUR"]

    # Define A,B,C = USD, EUR, JPY
    rate_ab = usd_to_eur      # USD -> EUR
    rate_bc = eur_to_jpy      # EUR -> JPY
    rate_ca = jpy_to_usd      # JPY -> USD

    return rate_ab, rate_bc, rate_ca


# ---------------------------------------------------------
# 4. AI MODEL TO PREDICT ARBITRAGE
# ---------------------------------------------------------

def generate_random_rates(n_samples=1000):
    """
    Generates random A->B, B->C, C->A rates.
    Returns feature matrix X and labels y.
    """
    data = []
    labels = []

    for _ in range(n_samples):
        rate_ab = random.uniform(0.5, 1.5)
        rate_bc = random.uniform(0.5, 1.5)
        rate_ca = random.uniform(0.5, 1.5)

        exists, profit = has_triangular_arbitrage(rate_ab, rate_bc, rate_ca)

        features = [
            rate_ab,
            rate_bc,
            rate_ca,
            rate_ab * rate_bc * rate_ca,  # product
            abs(rate_ab - rate_bc),
            abs(rate_bc - rate_ca),
            abs(rate_ca - rate_ab),
        ]

        data.append(features)
        labels.append(1 if exists else 0)

    return np.array(data), np.array(labels)


def train_arbitrage_classifier():
    """
    Trains a logistic regression classifier.
    """
    X, y = generate_random_rates(n_samples=2000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Model test accuracy: {acc:.3f}")

    return model


def predict_arbitrage(model, rate_ab, rate_bc, rate_ca):
    """
    Predicts arbitrage using the trained model.
    """
    features = [
        rate_ab,
        rate_bc,
        rate_ca,
        rate_ab * rate_bc * rate_ca,
        abs(rate_ab - rate_bc),
        abs(rate_bc - rate_ca),
        abs(rate_ca - rate_ab),
    ]

    X = np.array(features).reshape(1, -1)
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]

    return pred, prob


# ---------------------------------------------------------
# 5. MAIN DEMO (REAL DATA + AI + GRAPH)
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Training model on simulated data...")
    model = train_arbitrage_classifier()

    print("\nFetching real FX data for USD, EUR, JPY...")
    rate_ab, rate_bc, rate_ca = build_triangular_from_real_fx()
    print("Real rates:")
    print("USD -> EUR:", rate_ab)
    print("EUR -> JPY:", rate_bc)
    print("JPY -> USD:", rate_ca)

    exists, profit = has_triangular_arbitrage(rate_ab, rate_bc, rate_ca)
    print("\nSimple triangular arbitrage check:")
    print("Arbitrage exists:", exists, "Profit factor:", profit)

    pred, prob = predict_arbitrage(model, rate_ab, rate_bc, rate_ca)
    print("\nModel prediction on real data:")
    print("Model predicts arbitrage:", bool(pred))
    print("Probability:", prob)

    # Graph-based check using real rates
    rates_graph = {
        'USD': {'EUR': rate_ab, 'JPY': rate_bc * rate_ca},
        'EUR': {'USD': 1.0 / rate_ab, 'JPY': rate_bc},
        'JPY': {'USD': rate_ca, 'EUR': 1.0 / rate_bc}
    }

    print("\nGraph-based arbitrage (Bellman–Ford) on real data:")
    print("Arbitrage via negative cycle:", has_graph_arbitrage(rates_graph))
