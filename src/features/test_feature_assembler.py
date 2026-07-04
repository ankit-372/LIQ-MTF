from src.features.feature_assembler import FeatureAssembler

assembler = FeatureAssembler()

candle = {
    "open": 100,
    "high": 110,
    "low": 95,
    "close": 105,
    "volume": 150,
}

flow = {
    "agg_trade_count": 120,
    "agg_volume": 220,
    "agg_vwap": 104.7,
    "buyer_maker_ratio": 0.46,
}

liq = {
    "liquidity_up_5m": 112,
    "liquidity_below_5m": 97,
    "liquidity_up_1h": 118,
    "liquidity_below_1h": 93,
    "liquidity_up_4h": 125,
    "liquidity_below_4h": 90,
    "liquidity_up_1d": 140,
    "liquidity_below_1d": 80,
    "nearest_liq_5m": 97,
    "nearest_liq_1h": 93,
    "nearest_liq_4h": 90,
    "nearest_liq_1d": 80,
}

features = assembler.assemble(
    candle,
    flow,
    liq,
)

print(features)
print(len(features))