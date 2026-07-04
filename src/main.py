"""
Main Entrypoint for LIQ-MTF.
Instantiates and connects every layer.
"""
import asyncio
from src.core.config import Config
from src.core.logger import Logger
from src.core.alerter import Alerter
from src.core.heartbeat import Heartbeat
from src.core import event_bus
from src.liquidity.liquidity_detector import LiquidityDetector

# ----- MOCKS FOR MISSING LAYERS (Person 1 & 2) -----
class MockWSManager:
    async def run(self):
        """Simulates running WebSockets"""
        while True:
            await asyncio.sleep(1)
            # Simulate pinging heartbeat so it doesn't fail immediately in tests
            event_bus.publish("SYSTEM_PING", {"component": "ws1"})
            event_bus.publish("SYSTEM_PING", {"component": "ws2"})
            event_bus.publish("SYSTEM_PING", {"component": "ws3"})
            event_bus.publish("SYSTEM_PING", {"component": "ws4"})

class MockAgent:
    async def run(self):
        """Simulates running the Decision Agent"""
        while True:
            await asyncio.sleep(60)

async def main():
    # 1. Initialize Configuration
    config = Config()
    
    # 2. Initialize L6 Observability (Person 3)
    alerter = Alerter(config.TELEGRAM_TOKEN, config.TELEGRAM_CHAT_ID)
    logger = Logger(config.LOGS_DIR)  # noqa: F841
    heartbeat = Heartbeat(alerter)
    
    # 3. Initialize L2/L3 (Person 3)
    liquidity_detector = LiquidityDetector()  # noqa: F841
    
    # 4. Initialize Mocks for L1 & L4/L5
    ws_manager = MockWSManager()
    agent = MockAgent()
    
    # Send startup alert
    await alerter.send("SYSTEM_STARTUP", "LIQ-MTF system initialized.")
    
    # Run all background tasks concurrently
    try:
        await asyncio.gather(
            ws_manager.run(),
            heartbeat.run(),
            agent.run()
        )
    except asyncio.CancelledError:
        pass
    except Exception as e:
        await alerter.send("SYSTEM_ERROR", f"Fatal error in main loop: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("System shutdown requested.")
