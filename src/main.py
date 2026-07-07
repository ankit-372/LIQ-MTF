"""
Main Entrypoint for LIQ-MTF.
Instantiates and connects every layer.
"""
import asyncio
import logging
from src.core.config import Config
from src.core.logger import Logger
from src.core.alerter import Alerter
from src.core.heartbeat import Heartbeat
from src.core import event_bus
from src.ingest.ws_manager import WSManager
from src.liquidity.liquidity_detector import LiquidityDetector
from src.features.feature_assembler import FeatureAssembler
from src.predictor import ModelPredictor, PredictorRunner
from src.agent import DecisionAgent
from src.portfolio import PortfolioTracker
from src.circuit_breaker import CircuitBreaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

async def main():
    # 1. Initialize Configuration
    config = Config()
    
    # 2. Initialize L6 Observability
    alerter = Alerter(config.TELEGRAM_TOKEN, config.TELEGRAM_CHAT_ID)
    logger_instance = Logger(config.LOGS_DIR)  # noqa: F841
    heartbeat = Heartbeat(alerter)
    
    # 3. Initialize Portfolio & Circuit Breaker
    portfolio = PortfolioTracker(initial_balance=10000.0)
    circuit_breaker = CircuitBreaker(portfolio)
    
    # 4. Initialize L3 Liquidity Map
    liquidity_detector = LiquidityDetector()
    
    # 5. Initialize L2 Feature Engine
    feature_assembler = FeatureAssembler()
    
    # 6. Initialize L1 WSManager (which instantiates CandleBuilder, OrderFlow, BookTracker)
    ws_manager = WSManager()
    
    # 7. Initialize L4 Model Predictor & Runner
    # Check if we are running in paper test mode or live mode to use the correct model dir
    model_dir = "tests/mock_models" if config.TRADING_MODE == "paper" else "models"
    predictor = ModelPredictor(
        model_dir=model_dir, 
        confidence_threshold=config.MIN_CONFIDENCE
    )
    predictor_runner = PredictorRunner(predictor)  # noqa: F841
    
    # 8. Initialize L5/L6 Decision Agent & Executor Coordinator
    agent = DecisionAgent(
        portfolio=portfolio,
        circuit_breaker=circuit_breaker,
        db_journal=config.JOURNAL_PATH,
        db_matcher="models/signals_history.db",
        base_size=config.BASE_POSITION_SIZE,
        mode=config.TRADING_MODE
    )  # noqa: F841
    
    # 9. Wire L1 Ingest output to L2 Feature Engine input dynamically
    def on_flow_ready(flow_data):
        # Retrieve the latest 5m candle from WSManager's CandleBuilder
        candle = ws_manager.candle_builder.get_live_candle("5m")
        if not candle:
            return
            
        # Convert Candle dataclass instance to dict format
        candle_dict = {
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "open_time": candle.open_time,
            "close_time": candle.close_time
        }
        
        # Query L3 Liquidity Detector features
        liq_features = liquidity_detector.get_features(candle.close)
        
        # Assemble all 45 features
        features = feature_assembler.assemble(
            candle=candle_dict,
            flow_snapshot=flow_data,
            liquidity_snapshot=liq_features
        )
        
        # Publish FEATURES_READY to Event Bus
        event_bus.publish("FEATURES_READY", {"features": features})

    # Subscribe to FLOW_SNAPSHOT_READY to trigger feature assembly
    event_bus.subscribe("FLOW_SNAPSHOT_READY", on_flow_ready)
    
    # Send startup alert
    await alerter.send("SYSTEM_STARTUP", "LIQ-MTF system initialized and starting...")
    
    # Run all background tasks concurrently
    try:
        await asyncio.gather(
            ws_manager.run(),
            heartbeat.run()
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
