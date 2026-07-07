from collections import deque
from dataclasses import dataclass
import math
import statistics


@dataclass
class IndicatorSnapshot:

    # Existing
    log_ret: float
    volatility_20: float
    sma_20: float
    sma_ratio: float
    dist_liq_up_5m: float
    dist_liq_below_5m: float

    # New
    ema_9: float
    ema_21: float

    rsi_14: float

    atr_14: float

    macd: float
    macd_signal: float
    macd_histogram: float

    bb_upper: float
    bb_middle: float
    bb_lower: float


class IndicatorEngine:

    def __init__(self, window=20):

        self.window = window

        self.close_history = deque(maxlen=300)
        self.high_history = deque(maxlen=300)
        self.low_history = deque(maxlen=300)

        self.log_return_history = deque(maxlen=300)

        self.ema9 = None
        self.ema21 = None

        self.macd_signal = None
        self.macd_history = deque(maxlen=9)
        self.gain_history = deque(maxlen=14)
        self.loss_history = deque(maxlen=14)
        self.true_range_history = deque(maxlen=300)

    def process_candle(
        self,
        close,
        high,
        low,
        liquidity_up_5m,
        liquidity_below_5m,
    ):

        previous_close = (
            self.close_history[-1]
            if len(self.close_history) > 0
            else None
        )

        self.close_history.append(close)
        self.high_history.append(high)
        self.low_history.append(low)

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
        # EMA(9)
        # ----------------------------------

        alpha9 = 2 / (9 + 1)

        if self.ema9 is None:
            self.ema9 = close
        else:
            self.ema9 = (
                alpha9 * close
                + (1 - alpha9) * self.ema9
            )

        # ----------------------------------
        # EMA(21)
        # ----------------------------------

        alpha21 = 2 / (21 + 1)

        if self.ema21 is None:
            self.ema21 = close
        else:
            self.ema21 = (
                alpha21 * close
                + (1 - alpha21) * self.ema21
            )

        # ----------------------------------
        # RSI(14)
        # ----------------------------------

        if previous_close is None:

            rsi_14 = 50.0

        else:

            change = close - previous_close

            gain = max(change, 0.0)
            loss = max(-change, 0.0)

            self.gain_history.append(gain)
            self.loss_history.append(loss)

            if len(self.gain_history) < 14:

                rsi_14 = 50.0

            else:

                avg_gain = sum(self.gain_history) / len(self.gain_history)
                avg_loss = sum(self.loss_history) / len(self.loss_history)

                if avg_loss == 0:

                    rsi_14 = 100.0

                else:

                    rs = avg_gain / avg_loss

                    rsi_14 = 100 - (100 / (1 + rs))

        # ----------------------------------
        # ATR(14)
        # ----------------------------------

        if previous_close is None:

            true_range = high - low

        else:

            true_range = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )

        self.true_range_history.append(true_range)

        recent_tr_14 = list(self.true_range_history)[-14:]
        atr_14 = sum(recent_tr_14) / len(recent_tr_14) if recent_tr_14 else 0.0

        # ----------------------------------
        # MACD
        # ----------------------------------

        macd = self.ema9 - self.ema21

        self.macd_history.append(macd)

        if self.macd_signal is None:

            self.macd_signal = macd

        else:

            alpha_signal = 2 / (9 + 1)

            self.macd_signal = (
                alpha_signal * macd
                + (1 - alpha_signal) * self.macd_signal
            )

        macd_histogram = macd - self.macd_signal


        # ----------------------------------
        # Bollinger Bands (20)
        # ----------------------------------

        bb_middle = sma_20

        if len(self.close_history) > 1:

            std = statistics.stdev(self.close_history)

        else:

            std = 0.0

        bb_upper = bb_middle + (2 * std)
        bb_lower = bb_middle - (2 * std)
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

            ema_9=self.ema9,
            ema_21=self.ema21,

            rsi_14=rsi_14,

            atr_14=atr_14,

            macd=macd,
            macd_signal=self.macd_signal,
            macd_histogram=macd_histogram,

            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
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
    
    