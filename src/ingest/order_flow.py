from dataclasses import dataclass


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

    def process_trade(self, payload):

        if "q" not in payload or "m" not in payload:
            return

        quantity = float(payload["q"])
        is_buyer_maker = payload["m"]

        if is_buyer_maker:
            self.sell_volume += quantity
        else:
            self.buy_volume += quantity

        if quantity >= self.large_trade_threshold:
            self.large_trades.append(
                {
                    "price": float(payload.get("p", 0)),
                    "quantity": quantity,
                    "buyer_maker": is_buyer_maker,
                }
            )

    def build_flow_snapshot(self):

        if self.buy_volume == 0.0 and self.sell_volume == 0.0:
            delta = 0.0
        else:
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

        event = {
            "event": "FLOW_SNAPSHOT_READY",
            "buy_volume": snapshot.buy_volume,
            "sell_volume": snapshot.sell_volume,
            "delta": snapshot.delta,
            "cvd": snapshot.cvd,
            "cvd_slope": snapshot.cvd_slope,
            "large_trades": self.large_trades.copy(),
        }

        self.reset_window()

        return event

    def reset_window(self):

        self.buy_volume = 0.0
        self.sell_volume = 0.0

        self.large_trades.clear()