from dataclasses import dataclass
from src.core import event_bus


@dataclass
class FlowSnapshot:
    buy_volume: float
    sell_volume: float
    delta: float
    cvd: float
    cvd_slope: float


class OrderFlow:

    def __init__(self, large_trade_threshold=5.0):

        self.buy_volume = 0.0
        self.sell_volume = 0.0

        self.cvd = 0.0
        self.previous_cvd = 0.0

        self.large_trade_threshold = large_trade_threshold

        self.large_trades = []

        # Aggregated trade statistics
        self.trade_count = 0
        self.total_volume = 0.0
        self.total_price_volume = 0.0

        self.buy_count = 0
        self.sell_count = 0

    def process_trade(self, payload):

        if "q" not in payload or "m" not in payload or "p" not in payload:
            return

        quantity = float(payload["q"])
        price = float(payload["p"])
        is_buyer_maker = payload["m"]

        # Aggregate statistics
        self.trade_count += 1
        self.total_volume += quantity
        self.total_price_volume += price * quantity

        # Buy / Sell volume
        if is_buyer_maker:

            self.sell_volume += quantity
            self.sell_count += 1

        else:

            self.buy_volume += quantity
            self.buy_count += 1

        # Large trade detection
        if quantity >= self.large_trade_threshold:

            self.large_trades.append(
                {
                    "price": price,
                    "quantity": quantity,
                    "buyer_maker": is_buyer_maker,
                }
            )

    def build_flow_snapshot(self):

        delta = self.buy_volume - self.sell_volume

        self.cvd += delta

        cvd_slope = self.cvd - self.previous_cvd

        self.previous_cvd = self.cvd

        return FlowSnapshot(
            buy_volume=self.buy_volume,
            sell_volume=self.sell_volume,
            delta=delta,
            cvd=self.cvd,
            cvd_slope=cvd_slope,
        )

    def publish_snapshot(self):

        snapshot = self.build_flow_snapshot()

        # VWAP
        if self.total_volume > 0:
            agg_vwap = self.total_price_volume / self.total_volume
        else:
            agg_vwap = 0.0

        # Buyer maker ratio
        total_trades = self.buy_count + self.sell_count

        if total_trades > 0:
            buyer_maker_ratio = self.buy_count / total_trades
        else:
            buyer_maker_ratio = 0.5

        event = {

            "event": "FLOW_SNAPSHOT_READY",

            "buy_volume": snapshot.buy_volume,
            "sell_volume": snapshot.sell_volume,

            "delta": snapshot.delta,

            "cvd": snapshot.cvd,
            "cvd_slope": snapshot.cvd_slope,

            # ML Features
            "agg_trade_count": self.trade_count,
            "agg_volume": self.total_volume,
            "agg_vwap": agg_vwap,
            "buyer_maker_ratio": buyer_maker_ratio,

            "large_trades": self.large_trades.copy(),

        }

        # Publish for downstream consumers
        event_bus.publish(
            "FLOW_SNAPSHOT_READY",
            event,
        )

        self.reset_window()

        return event

    def reset_window(self):

        # Window statistics
        self.buy_volume = 0.0
        self.sell_volume = 0.0

        self.trade_count = 0
        self.total_volume = 0.0
        self.total_price_volume = 0.0

        self.buy_count = 0
        self.sell_count = 0

        self.large_trades.clear()