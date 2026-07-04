from collections import deque
from dataclasses import dataclass
import math
import statistics


@dataclass
class IndicatorSnapshot:
    log_ret: float
    volatility_20: float
    sma_20: float
    sma_ratio: float
    dist_liq_up_5m: float
    dist_liq_below_5m: float


class IndicatorEngine:

    def __init__(self, window=20):

        self.window = window

        self.close_history = deque(maxlen=window)
        self.log_return_history = deque(maxlen=window)

    def process_candle(
        self,
        close,
        liquidity_up_5m,
        liquidity_below_5m,
    ):

        previous_close = (
            self.close_history[-1]
            if len(self.close_history) > 0
            else None
        )

        self.close_history.append(close)

        # ----------------------------------
        # Log Return
        # ----------------------------------

        if previous_close is None or previous_close <= 0:

            log_ret = 0.0

        else:

            log_ret = math.log(close / previous_close)

        self.log_return_history.append(log_ret)

        # ----------------------------------
        # SMA(20)
        # ----------------------------------

        sma_20 = (
            sum(self.close_history)
            / len(self.close_history)
        )

        if sma_20 == 0:

            sma_ratio = 1.0

        else:

            sma_ratio = close / sma_20

        # ----------------------------------
        # Rolling Volatility
        # ----------------------------------

        if len(self.log_return_history) > 1:

            volatility_20 = statistics.stdev(
                self.log_return_history
            )

        else:

            volatility_20 = 0.0

        # ----------------------------------
        # Liquidity Distances
        # ----------------------------------

        if close == 0:

            dist_liq_up = 0.0
            dist_liq_below = 0.0

        else:

            dist_liq_up = (
                liquidity_up_5m - close
            ) / close

            dist_liq_below = (
                close - liquidity_below_5m
            ) / close

        return IndicatorSnapshot(

            log_ret=log_ret,

            volatility_20=volatility_20,

            sma_20=sma_20,

            sma_ratio=sma_ratio,

            dist_liq_up_5m=dist_liq_up,

            dist_liq_below_5m=dist_liq_below,

        )

    def get_latest(self):

        if len(self.close_history) == 0:

            return None

        close = self.close_history[-1]

        sma_20 = (
            sum(self.close_history)
            / len(self.close_history)
        )

        return {

            "close": close,

            "sma_20": sma_20,

            "history": len(self.close_history),

        }