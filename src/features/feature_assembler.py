from src.shared.feature_registry import FEATURE_COLUMNS
from src.core.event_bus import publish
from .indicator_engine import IndicatorEngine


class FeatureAssembler:

    def __init__(self):

        self.indicator_engine = IndicatorEngine()

    def assemble(
        self,
        candle,
        flow_snapshot,
        liquidity_snapshot,
    ):

        indicators = self.indicator_engine.process_candle(

            close=candle["close"],

            liquidity_up_5m=liquidity_snapshot["liquidity_up_5m"],

            liquidity_below_5m=liquidity_snapshot["liquidity_below_5m"],

        )

        features = {

            "agg_trade_count":
                flow_snapshot.get("agg_trade_count", 0),

            "agg_volume":
                flow_snapshot.get("agg_volume", 0.0),

            "agg_vwap":
                flow_snapshot.get("agg_vwap", candle["close"]),

            "buyer_maker_ratio":
                flow_snapshot.get("buyer_maker_ratio", 0.5),

            "open":
                candle["open"],

            "high":
                candle["high"],

            "low":
                candle["low"],

            "close":
                candle["close"],

            "volume":
                candle["volume"],

            "quote_volume":
                candle.get("quote_volume", 0.0),

            "count":
                candle.get("count", 0),

            "taker_buy_volume":
                candle.get("taker_buy_volume", 0.0),

            "taker_buy_quote_volume":
                candle.get("taker_buy_quote_volume", 0.0),

            "liquidity_up_5m":
                liquidity_snapshot.get("liquidity_up_5m", 0.0),

            "liquidity_below_5m":
                liquidity_snapshot.get("liquidity_below_5m", 0.0),

            "liquidity_up_1h":
                liquidity_snapshot.get("liquidity_up_1h", 0.0),

            "liquidity_below_1h":
                liquidity_snapshot.get("liquidity_below_1h", 0.0),

            "liquidity_up_4h":
                liquidity_snapshot.get("liquidity_up_4h", 0.0),

            "liquidity_below_4h":
                liquidity_snapshot.get("liquidity_below_4h", 0.0),

            "liquidity_up_1d":
                liquidity_snapshot.get("liquidity_up_1d", 0.0),

            "liquidity_below_1d":
                liquidity_snapshot.get("liquidity_below_1d", 0.0),

            "nearest_liq_5m":
                liquidity_snapshot.get("nearest_liq_5m", 0.0),

            "nearest_liq_1h":
                liquidity_snapshot.get("nearest_liq_1h", 0.0),

            "nearest_liq_4h":
                liquidity_snapshot.get("nearest_liq_4h", 0.0),

            "nearest_liq_1d":
                liquidity_snapshot.get("nearest_liq_1d", 0.0),

            "log_ret":
                indicators.log_ret,

            "volatility_20":
                indicators.volatility_20,

            "sma_ratio":
                indicators.sma_ratio,

            "dist_liq_up_5m":
                indicators.dist_liq_up_5m,

            "dist_liq_below_5m":
                indicators.dist_liq_below_5m,

        }

        ordered_features = {

            feature: features.get(feature)

            for feature in FEATURE_COLUMNS

        }

        publish(

            "FEATURES_READY",

            ordered_features,

        )

        return ordered_features