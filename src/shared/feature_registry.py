"""
Feature Registry Module
Defines the official feature set and ordering expected by the Level 4 LightGBM model.
"""

FEATURE_COLUMNS = [
    # 1. Technical Indicators
    "pa_atr_5m",
    "pa_atr_1h",
    "pa_atr_4h",
    "pa_rsi_14_5m",
    "pa_rsi_14_1h",
    "pa_ema_9_5m",
    "pa_ema_21_5m",
    "pa_ema_cross_5m",
    "pa_ema_9_1h",
    "pa_ema_21_1h",
    "pa_ema_cross_1h",
    "pa_macd_hist_5m",
    "pa_macd_hist_1h",
    "pa_boll_pct_5m",
    "pa_return_5m",
    "pa_return_1h",

    # 2. Order Flow Metrics
    "flow_delta_5m",
    "flow_cvd_slope",
    "flow_buy_sell_ratio",
    "flow_large_trade_ratio",
    "flow_buy_volume",
    "flow_sell_volume",

    # 3. Liquidity Levels
    "lvl_dist_above_5m",
    "lvl_dist_below_5m",
    "lvl_dist_above_1h",
    "lvl_dist_below_1h",
    "lvl_dist_above_4h",
    "lvl_dist_below_4h",
    "lvl_active_count",
    "lvl_recent_sweep_count",
    "lvl_nearest_confluence",

    # 4. Volatility & Calendar
    "vol_atr_ratio",
    "vol_regime",
    "cal_hour_sin",
    "cal_hour_cos",
    "cal_day_sin",
    "cal_day_cos",

    # 5. Raw Candle Columns
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_volume",
    "taker_buy_quote_volume"
]
