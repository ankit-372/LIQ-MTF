from dataclasses import dataclass


@dataclass
class Candle:

    interval: str

    open_time: int
    close_time: int

    open: float
    high: float
    low: float
    close: float

    volume: float


class CandleBuilder:

    def __init__(self):

        self.live_candles = {

            "5m": None,
            "1h": None,
            "4h": None,

        }

    def process_kline(self, interval, payload):

        kline = payload.get("k")

        if not kline:
            return None

        candle = Candle(

            interval=interval,

            open_time=kline["t"],
            close_time=kline["T"],

            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),

            volume=float(kline["v"]),

        )

        # Store latest live candle
        self.live_candles[interval] = candle

        # Publish only when candle is closed
        if not kline.get("x", False):
            return None

        event = {

            "event": f"CANDLE_CLOSED_{interval.upper()}",
            "interval": interval,

            "open_time": candle.open_time,
            "close_time": candle.close_time,

            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,

            "volume": candle.volume,

        }

        return event

    def get_live_candle(self, interval):

        return self.live_candles.get(interval)