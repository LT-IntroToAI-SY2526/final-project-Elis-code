import numpy as np
from abides_core import NanoTime, SimulationConfig
from abides_markets.agents import (
    ExchangeAgent,
    MarketMakerAgent,
    NoiseAgent
)
from abides_markets.markets import OrderBook


def generate_config():
    """
    Three-exchange ABIDES simulation with built-in arbitrage.
    Exchanges:
      EX1 = "binance"
      EX2 = "coinbase"
      EX3 = "kraken"
    """

    # -------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------
    start_time = NanoTime("09:30:00")
    end_time   = NanoTime("16:00:00")
    seed       = 42

    # Base midprice for all exchanges
    base_price = 30000

    # -------------------------------------------------------
    # Exchange definitions
    # -------------------------------------------------------
    exchanges = []

    # Exchange 1: Binance-like (slightly cheaper)
    exchanges.append(
        ExchangeAgent(
            id=1,
            name="EX1",
            starting_cash=0,
            log_orders=True,
            book=OrderBook(
                symbol="BTC",
                starting_price=base_price - 10
            )
        )
    )

    # Exchange 2: Coinbase-like (neutral)
    exchanges.append(
        ExchangeAgent(
            id=2,
            name="EX2",
            starting_cash=0,
            log_orders=True,
            book=OrderBook(
                symbol="BTC",
                starting_price=base_price
            )
        )
    )

    # Exchange 3: Kraken-like (slightly more expensive)
    exchanges.append(
        ExchangeAgent(
            id=3,
            name="EX3",
            starting_cash=0,
            log_orders=True,
            book=OrderBook(
                symbol="BTC",
                starting_price=base_price + 10
            )
        )
    )

    # -------------------------------------------------------
    # Market Makers (provide liquidity)
    # -------------------------------------------------------
    market_makers = []
    mm_id = 10

    for ex in ["EX1", "EX2", "EX3"]:
        market_makers.append(
            MarketMakerAgent(
                id=mm_id,
                name=f"MM_{ex}",
                symbol="BTC",
                starting_cash=1_000_000,
                exchange_id=ex,
                r_bar=base_price,
                kappa=0.05,
                lambda_a=0.01,
                lambda_b=0.01,
                sigma_n=0.1,
                q_max=10
            )
        )
        mm_id += 1

    # -------------------------------------------------------
    # Noise Traders (random order flow)
    # -------------------------------------------------------
    noise_traders = []
    nt_id = 100

    for _ in range(50):
        noise_traders.append(
            NoiseAgent(
                id=nt_id,
                name=f"NT_{nt_id}",
                symbol="BTC",
                starting_cash=100_000,
                exchange_ids=["EX1", "EX2", "EX3"],
                order_size=1,
                arrival_rate=0.1
            )
        )
        nt_id += 1

    # -------------------------------------------------------
    # Combine all agents
    # -------------------------------------------------------
    agents = exchanges + market_makers + noise_traders

    # -------------------------------------------------------
    # Build simulation config
    # -------------------------------------------------------
    return SimulationConfig(
        agents=agents,
        start_time=start_time,
        end_time=end_time,
        seed=seed,
        log_dir="abides_logs"
    )


config = generate_config()
