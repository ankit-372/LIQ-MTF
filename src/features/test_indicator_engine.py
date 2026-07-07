from src.features.indicator_engine import IndicatorEngine

engine = IndicatorEngine()

prices = [
    100,
    101,
    102,
    101,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
]

for price in prices:

    snapshot = engine.process_candle(
        close=118,
        high=120,
        low=116,
        liquidity_up_5m=123,
        liquidity_below_5m=113,
    )

print(snapshot)