"""
Mock payloads shared across the project for integration.

These payloads define the expected event contracts
between the Data Ingest Layer (L1) and downstream modules.
"""

# =====================================================
# Candle Closed (5m)
# =====================================================

CANDLE_CLOSED_5M = {

    "event": "CANDLE_CLOSED_5M",

    "open_time": 1719000000000,

    "close_time": 1719000300000,

    "timeframe": "5m",

    "open": 75400.5,
    "high": 75600.0,
    "low": 75350.2,
    "close": 75500.0,

    "volume": 125.345,

    "quote_volume": 9465312.43,

    "count": 284,

    "taker_buy_volume": 72.15,

    "taker_buy_quote_volume": 5443312.25,

}

# =====================================================
# Flow Snapshot
# =====================================================

FLOW_SNAPSHOT_READY = {

    "event": "FLOW_SNAPSHOT_READY",

    "buy_volume": 67.4,

    "sell_volume": 51.1,

    "delta": 16.3,

    "cvd": 1840.3,

    "cvd_slope": 16.3,

    "agg_trade_count": 312,

    "agg_volume": 118.5,

    "agg_vwap": 75486.72,

    "buyer_maker_ratio": 0.57,

    "large_trades": [

        {

            "price": 75510.2,

            "quantity": 7.5,

            "buyer_maker": False,

        }

    ],

}