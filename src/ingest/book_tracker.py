from dataclasses import dataclass


@dataclass
class BookSnapshot:

    bid: float
    ask: float
    spread: float
    spread_bps: float


class BookTracker:

    def __init__(self):

        self.bid = 0.0
        self.ask = 0.0

        self.spread = 0.0
        self.spread_bps = 0.0

    def process_book(self, payload):

        if "b" not in payload or "a" not in payload:
            return

        self.bid = float(payload["b"])
        self.ask = float(payload["a"])

        self.spread = self.ask - self.bid

        mid_price = (self.ask + self.bid) / 2

        if mid_price == 0:

            self.spread_bps = 0.0

        else:

            self.spread_bps = (
                self.spread / mid_price
            ) * 10000

    def get_snapshot(self):

        return BookSnapshot(

            bid=self.bid,
            ask=self.ask,
            spread=self.spread,
            spread_bps=self.spread_bps,

        )

    def publish_snapshot(self):

        snapshot = self.get_snapshot()

        event = {

            "event": "BOOK_SNAPSHOT_READY",
            "bid": snapshot.bid,
            "ask": snapshot.ask,
            "spread": snapshot.spread,
            "spread_bps": snapshot.spread_bps,

        }

        return event