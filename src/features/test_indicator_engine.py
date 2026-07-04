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

        close=price,

        liquidity_up_5m=price + 5,

        liquidity_below_5m=price - 5,

    )

print(snapshot)