import math
import time
from src.shared.feature_registry import FEATURE_COLUMNS
from src.core import event_bus
from .indicator_engine import IndicatorEngine


class FeatureAssembler:

    def __init__(self):

        self.indicator_5m = IndicatorEngine()
        self.indicator_1h = IndicatorEngine()
        self.indicator_4h = IndicatorEngine()

        self.last_snapshot_1h = None
        self.last_snapshot_4h = None

        # Subscribe to Event Bus for higher timeframes
        event_bus.subscribe("CANDLE_CLOSED_1H", self.on_1h_candle)
        event_bus.subscribe("CANDLE_CLOSED_4H", self.on_4h_candle)

    def on_1h_candle(self, event):
        self.last_snapshot_1h = self.indicator_1h.process_candle(
            close=event["close"],
            high=event["high"],
            low=event["low"],
            liquidity_up_5m=event["close"],
            liquidity_below_5m=event["close"]
        )

    def on_4h_candle(self, event):
        self.last_snapshot_4h = self.indicator_4h.process_candle(
            close=event["close"],
            high=event["high"],
            low=event["low"],
            liquidity_up_5m=event["close"],
            liquidity_below_5m=event["close"]
        )

    def assemble(
        self,
        candle,
        flow_snapshot,
        liquidity_snapshot,
    ):

        indicators_5m = self.indicator_5m.process_candle(
            close=candle["close"],
            high=candle["high"],
            low=candle["low"],
            liquidity_up_5m=liquidity_snapshot.get("liquidity_up_5m", candle["close"]),
            liquidity_below_5m=liquidity_snapshot.get("liquidity_below_5m", candle["close"]),
        )

        # 1h indicators fallback logic
        if self.last_snapshot_1h:
            atr_1h = self.last_snapshot_1h.atr_14
            rsi_1h = self.last_snapshot_1h.rsi_14
            ema_9_1h = self.last_snapshot_1h.ema_9
            ema_21_1h = self.last_snapshot_1h.ema_21
            macd_hist_1h = self.last_snapshot_1h.macd_histogram
            return_1h = self.last_snapshot_1h.log_ret
        else:
            atr_1h = indicators_5m.atr_14
            rsi_1h = indicators_5m.rsi_14
            ema_9_1h = indicators_5m.ema_9
            ema_21_1h = indicators_5m.ema_21
            macd_hist_1h = indicators_5m.macd_histogram
            return_1h = indicators_5m.log_ret

        # 4h indicators fallback logic
        if self.last_snapshot_4h:
            atr_4h = self.last_snapshot_4h.atr_14
        else:
            atr_4h = indicators_5m.atr_14

        # Volatility & Average ATR calculation (24h = 288 periods of 5m)
        recent_tr_288 = list(self.indicator_5m.true_range_history)[-288:]
        avg_tr_24h = sum(recent_tr_288) / len(recent_tr_288) if recent_tr_288 else 0.0
        
        current_atr_5m = indicators_5m.atr_14
        vol_atr_ratio = current_atr_5m / avg_tr_24h if avg_tr_24h != 0 else 1.0

        # vol_regime encoding (0 = low, 1 = normal, 2 = high, 3 = extreme)
        if vol_atr_ratio < 0.8:
            vol_regime = 0.0
        elif vol_atr_ratio < 1.2:
            vol_regime = 1.0
        elif vol_atr_ratio < 2.0:
            vol_regime = 2.0
        else:
            vol_regime = 3.0

        # Calendar features (sin/cos of hour and day of week)
        open_time = candle.get("open_time", int(time.time() * 1000))
        timestamp_s = open_time / 1000.0
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
        
        hour = dt.hour + dt.minute / 60.0
        cal_hour_sin = math.sin(2 * math.pi * hour / 24.0)
        cal_hour_cos = math.cos(2 * math.pi * hour / 24.0)
        
        day = dt.weekday() + hour / 24.0
        cal_day_sin = math.sin(2 * math.pi * day / 7.0)
        cal_day_cos = math.cos(2 * math.pi * day / 7.0)

        # Order Flow mapping
        trade_count = flow_snapshot.get("agg_trade_count", 0)
        large_trades = flow_snapshot.get("large_trades", [])
        flow_large_trade_ratio = len(large_trades) / trade_count if trade_count > 0 else 0.0

        features = {
            # 1. Technical Indicators
            "pa_atr_5m": current_atr_5m,
            "pa_atr_1h": atr_1h,
            "pa_atr_4h": atr_4h,
            "pa_rsi_14_5m": indicators_5m.rsi_14,
            "pa_rsi_14_1h": rsi_1h,
            "pa_ema_9_5m": indicators_5m.ema_9,
            "pa_ema_21_5m": indicators_5m.ema_21,
            "pa_ema_cross_5m": 1.0 if indicators_5m.ema_9 > indicators_5m.ema_21 else 0.0,
            "pa_ema_9_1h": ema_9_1h,
            "pa_ema_21_1h": ema_21_1h,
            "pa_ema_cross_1h": 1.0 if ema_9_1h > ema_21_1h else 0.0,
            "pa_macd_hist_5m": indicators_5m.macd_histogram,
            "pa_macd_hist_1h": macd_hist_1h,
            "pa_boll_pct_5m": (candle["close"] - indicators_5m.bb_lower) / (indicators_5m.bb_upper - indicators_5m.bb_lower) if (indicators_5m.bb_upper - indicators_5m.bb_lower) != 0 else 0.5,
            "pa_return_5m": indicators_5m.log_ret,
            "pa_return_1h": return_1h,

            # 2. Order Flow Metrics
            "flow_delta_5m": flow_snapshot.get("delta", 0.0),
            "flow_cvd_slope": flow_snapshot.get("cvd_slope", 0.0),
            "flow_buy_sell_ratio": flow_snapshot.get("buyer_maker_ratio", 0.5),
            "flow_large_trade_ratio": flow_large_trade_ratio,
            "flow_buy_volume": flow_snapshot.get("buy_volume", 0.0),
            "flow_sell_volume": flow_snapshot.get("sell_volume", 0.0),

            # 3. Liquidity Levels
            "lvl_dist_above_5m": liquidity_snapshot.get("lvl_dist_above_5m", liquidity_snapshot.get("dist_liq_up_5m", (liquidity_snapshot.get("nearest_liq_5m", candle["close"]) - candle["close"]) / candle["close"])),
            "lvl_dist_below_5m": liquidity_snapshot.get("lvl_dist_below_5m", liquidity_snapshot.get("dist_liq_below_5m", (candle["close"] - liquidity_snapshot.get("nearest_liq_5m", candle["close"])) / candle["close"])),
            "lvl_dist_above_1h": liquidity_snapshot.get("lvl_dist_above_1h", (liquidity_snapshot.get("nearest_liq_1h", candle["close"]) - candle["close"]) / candle["close"] if "nearest_liq_1h" in liquidity_snapshot else 999.0),
            "lvl_dist_below_1h": liquidity_snapshot.get("lvl_dist_below_1h", (candle["close"] - liquidity_snapshot.get("nearest_liq_1h", candle["close"])) / candle["close"] if "nearest_liq_1h" in liquidity_snapshot else 999.0),
            "lvl_dist_above_4h": liquidity_snapshot.get("lvl_dist_above_4h", (liquidity_snapshot.get("nearest_liq_4h", candle["close"]) - candle["close"]) / candle["close"] if "nearest_liq_4h" in liquidity_snapshot else 999.0),
            "lvl_dist_below_4h": liquidity_snapshot.get("lvl_dist_below_4h", (candle["close"] - liquidity_snapshot.get("nearest_liq_4h", candle["close"])) / candle["close"] if "nearest_liq_4h" in liquidity_snapshot else 999.0),
            "lvl_active_count": liquidity_snapshot.get("lvl_active_count", 0.0),
            "lvl_recent_sweep_count": liquidity_snapshot.get("lvl_recent_sweep_count", 0.0),
            "lvl_nearest_confluence": liquidity_snapshot.get("lvl_nearest_confluence", 0.0),

            # 4. Volatility & Calendar
            "vol_atr_ratio": vol_atr_ratio,
            "vol_regime": vol_regime,
            "cal_hour_sin": cal_hour_sin,
            "cal_hour_cos": cal_hour_cos,
            "cal_day_sin": cal_day_sin,
            "cal_day_cos": cal_day_cos,

            # 5. Raw Candle Columns
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
            "quote_volume": candle.get("quote_volume", 0.0),
            "taker_buy_volume": candle.get("taker_buy_volume", 0.0),
            "taker_buy_quote_volume": candle.get("taker_buy_quote_volume", 0.0)
        }

        # Filter and order the dict based on FEATURE_COLUMNS registry
        ordered_features = {
            feature: features.get(feature, 0.0)
            for feature in FEATURE_COLUMNS
        }

        event_bus.publish(
            "FEATURES_READY",
            ordered_features,
        )

        return ordered_features