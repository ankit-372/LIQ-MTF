import time
import numpy as np
from types import MappingProxyType
from src.shared.event_bus import EventBus

class ScenarioManager:
    def __init__(self):
        # Subscribe to ML_SIGNAL_GENERATED event
        EventBus.subscribe("ML_SIGNAL_GENERATED", self.on_ml_signal_generated)
        print("ScenarioManager: Subscribed to 'ML_SIGNAL_GENERATED'")

    def package_scenario(self, market_tick: dict, ml_pred: dict) -> MappingProxyType:
        """
        Aggregates price feed, technical indicators, order book liquidity maps,
        and machine learning signals into a frozen 25-field scenario snapshot dict.
        """
        close_p = float(market_tick.get("close", 0.0))
        sma_20 = float(market_tick.get("sma_20", close_p))
        
        # Calculate trend on-the-fly if not provided
        trend = "BULLISH" if close_p >= sma_20 else "BEARISH"
        
        raw_dict = {
            # 1. Temporal marker
            "timestamp": market_tick.get("timestamp", int(time.time())),
            
            # 2. Base Price OHLC
            "open": float(market_tick.get("open", 0.0)),
            "high": float(market_tick.get("high", 0.0)),
            "low": float(market_tick.get("low", 0.0)),
            "close": close_p,
            "volume": float(market_tick.get("volume", 0.0)),
            
            # 3. Aggregated Trade Flow Indicators
            "quote_volume": float(market_tick.get("quote_volume", 0.0)),
            "count": int(market_tick.get("count", 0)),
            "taker_buy_volume": float(market_tick.get("taker_buy_volume", 0.0)),
            "agg_trade_count": int(market_tick.get("agg_trade_count", 0)),
            "agg_volume": float(market_tick.get("agg_volume", 0.0)),
            "agg_vwap": float(market_tick.get("agg_vwap", close_p)),
            "buyer_maker_ratio": float(market_tick.get("buyer_maker_ratio", 0.5)),
            
            # 4. Volatility and Trend Indicators
            "volatility_20": float(market_tick.get("volatility_20", 0.0)),
            "sma_ratio": float(market_tick.get("sma_ratio", 1.0)),
            "market_trend": trend,
            
            # 5. Order Book Liquidity Limits
            "liquidity_up_5m": float(market_tick.get("liquidity_up_5m", 0.0)),
            "liquidity_below_5m": float(market_tick.get("liquidity_below_5m", 0.0)),
            "nearest_liq_5m": float(market_tick.get("nearest_liq_5m", 0.0)),
            "nearest_liq_1h": float(market_tick.get("nearest_liq_1h", 0.0)),
            "nearest_liq_4h": float(market_tick.get("nearest_liq_4h", 0.0)),
            "nearest_liq_1d": float(market_tick.get("nearest_liq_1d", 0.0)),
            
            # 6. Normalized Liquidity Boundaries
            "dist_liq_up_5m": float(market_tick.get("dist_liq_up_5m", 0.0)),
            "dist_liq_below_5m": float(market_tick.get("dist_liq_below_5m", 0.0)),
            
            # 7. Machine Learning Predictions
            "ml_signal": str(ml_pred.get("signal", "WAIT")),
            "ml_confidence": float(ml_pred.get("confidence", 0.0))
        }
        
        # Verify we packaged at least 20+ fields
        assert len(raw_dict) >= 20, f"Error: Scenario contains only {len(raw_dict)} fields, expected >= 20."
        
        # Return a frozen snapshot (MappingProxyType enforces read-only attributes)
        return MappingProxyType(raw_dict)

    def on_ml_signal_generated(self, data: dict):
        """
        Triggered when ML signals are published. Packages the snapshot
        and republishes the frozen scenario for subsequent pipeline stages.
        """
        market_tick = data.get("market_tick", {})
        ml_pred = data.get("ml_prediction", {})
        
        # Build and freeze the scenario snapshot
        frozen_scenario = self.package_scenario(market_tick, ml_pred)
        
        # Publish scenario to event bus for pattern matching and risk rules evaluation
        EventBus.publish("SCENARIO_CREATED", {"scenario": frozen_scenario})
