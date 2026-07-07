import asyncio
import logging
import time

import orjson
import websockets
from websockets.exceptions import ConnectionClosed

from src.ingest.candle_builder import CandleBuilder
from src.ingest.order_flow import OrderFlow
from src.ingest.book_tracker import BookTracker
from src.core import event_bus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


class WSManager:

    def __init__(self):

        self.candle_builder = CandleBuilder()
        self.order_flow = OrderFlow()
        self.book_tracker = BookTracker()

        self.stream_map = {
            "aggTrade": "btcusdt@aggTrade",
            "bookTicker": "btcusdt@bookTicker",
            "5m": "btcusdt@kline_5m",
            "1h": "btcusdt@kline_1h",
            "4h": "btcusdt@kline_4h",
        }

        self.urls = {
            "aggTrade":
                "wss://fstream.binance.com/ws/btcusdt@aggTrade",

            "bookTicker":
                "wss://fstream.binance.com/ws/btcusdt@bookTicker",

            "5m":
                "wss://fstream.binance.com/ws/btcusdt@kline_5m",

            "1h_4h":
                "wss://fstream.binance.com/stream?streams=btcusdt@kline_1h/btcusdt@kline_4h",
        }

    # ==================================================
    # Message Handlers
    # ==================================================

    def handle_trade(self, payload):

        self.order_flow.process_trade(payload)


    def handle_book_ticker(self, payload):

        self.book_tracker.process_book(payload)

        book_event = self.book_tracker.publish_snapshot()

        logging.info(book_event)


    def handle_kline_5m(self, payload):

        candle_event = self.candle_builder.process_kline(
            "5m",
            payload
        )

        if candle_event:

            logging.info(candle_event)

            flow_event = self.order_flow.publish_snapshot()

            logging.info(flow_event)


    def handle_kline_1h(self, payload):

        candle_event = self.candle_builder.process_kline(
            "1h",
            payload
        )

        if candle_event:

            logging.info(candle_event)


    def handle_kline_4h(self, payload):

        candle_event = self.candle_builder.process_kline(
            "4h",
            payload
        )

        if candle_event:

            logging.info(candle_event)

    # ==================================================
    # Message Router
    # ==================================================

    def route_message(self, stream, payload):
        self.publish_ping(stream)

        if stream == "btcusdt@aggTrade":

            self.handle_trade(payload)

        elif stream == "btcusdt@bookTicker":

            self.handle_book_ticker(payload)

        elif stream == "btcusdt@kline_5m":

            self.handle_kline_5m(payload)

        elif stream == "btcusdt@kline_1h":

            self.handle_kline_1h(payload)

        elif stream == "btcusdt@kline_4h":

            self.handle_kline_4h(payload)

        else:

            logging.warning(f"Unknown stream: {stream}")

    def publish_ping(self, stream):
        component = None
        if stream == "btcusdt@aggTrade":
            component = "ws1"
        elif stream == "btcusdt@kline_5m":
            component = "ws2"
        elif stream == "btcusdt@bookTicker":
            component = "ws3"
        elif stream in ["btcusdt@kline_1h", "btcusdt@kline_4h"]:
            component = "ws4"

        if component:
            event_bus.publish("SYSTEM_PING", {"component": component})

    # ==================================================
    # WebSocket Connection
    # ==================================================

    async def connect(self, stream_name, url):

        retry_delay = 1

        while True:

            try:

                logging.info("=" * 70)
                logging.info(f"Connecting [{stream_name}] -> {url}")

                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as websocket:

                    logging.info(f"{stream_name} : WS_RECONNECTED")
                    event_bus.publish("WS_RECONNECTED", {"stream": stream_name})

                    retry_delay = 1

                    while True:

                        try:

                            message = await websocket.recv()

                            payload = orjson.loads(message)

                            # Handle combined stream format
                            if isinstance(payload, dict) and "stream" in payload and "data" in payload:
                                stream = payload["stream"]
                                data = payload["data"]
                            else:
                                stream = self.stream_map.get(stream_name)
                                data = payload

                            # Calculate and publish latency
                            event_time = data.get("E") if isinstance(data, dict) else None
                            if event_time is None and isinstance(data, dict) and "k" in data:
                                event_time = data["k"].get("t")

                            if event_time:
                                latency_ms = int(time.time() * 1000) - int(event_time)
                                event_bus.publish("WS_HEARTBEAT", {"latency_ms": max(0, latency_ms)})
                            else:
                                event_bus.publish("WS_HEARTBEAT", {"latency_ms": 0})

                            self.route_message(
                                stream,
                                data
                            )

                        except ConnectionClosed as e:

                            logging.warning(
                                f"{stream_name} : WS_DISCONNECTED : {e}"
                            )
                            event_bus.publish("WS_DISCONNECTED", {"stream": stream_name})

                            break

            except Exception as e:

                logging.exception(
                    f"{stream_name} : Unexpected Error : {e}"
                )
                event_bus.publish("WS_DISCONNECTED", {"stream": stream_name})

            logging.info(
                f"{stream_name} reconnecting in {retry_delay} seconds..."
            )

            await asyncio.sleep(retry_delay)

            retry_delay = min(
                retry_delay * 2,
                30
            )

    # ==================================================
    # Start
    # ==================================================

    async def start(self):

        tasks = [

            asyncio.create_task(
                self.connect(stream_name, url)
            )

            for stream_name, url in self.urls.items()

        ]

        await asyncio.gather(*tasks)


# ==================================================
# Main
# ==================================================

async def main():

    manager = WSManager()

    await manager.start()


if __name__ == "__main__":

    asyncio.run(main())