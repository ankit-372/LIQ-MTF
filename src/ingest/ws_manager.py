import asyncio
import logging

import orjson
import websockets
from websockets.exceptions import ConnectionClosed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

BASE_URL = "wss://stream.binancefuture.com/ws"
STREAM = "btcusdt@aggTrade"


class WSManager:

    def __init__(self):
        self.url = f"{BASE_URL}/{STREAM}"

    async def start(self):

        retry_delay = 1

        while True:

            try:

                logging.info("=" * 60)
                logging.info(f"Connecting to {self.url}")

                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as websocket:

                    logging.info("✅ WS_RECONNECTED")

                    # Connection successful
                    retry_delay = 1

                    while True:

                        try:

                            message = await websocket.recv()

                            data = orjson.loads(message)

                            # Phase 1
                            # Later this will be routed to candle_builder,
                            # order_flow and book_tracker
                            print(data)

                        except ConnectionClosed:

                            logging.warning(
                                "WebSocket connection closed."
                            )

                            break

            except Exception as e:

                logging.error(f"WS_DISCONNECTED : {e}")

            logging.info(
                f"Reconnecting in {retry_delay} seconds..."
            )

            await asyncio.sleep(retry_delay)

            retry_delay = min(retry_delay * 2, 30)


async def main():

    manager = WSManager()

    await manager.start()


if __name__ == "__main__":
    asyncio.run(main())