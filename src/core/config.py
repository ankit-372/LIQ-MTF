"""
Centralized Configuration for LIQ-MTF.
Loads all settings from environment variables with sensible defaults.
"""
import os
from typing import Optional

# Try loading from .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Config:
    # Trading Configuration
    TRADING_MODE: str = os.getenv("TRADING_MODE", "paper")  # paper / testnet / live
    BASE_POSITION_SIZE: float = float(os.getenv("BASE_POSITION_SIZE", "0.003"))
    MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.65"))
    
    # Risk Management
    SL_MULTIPLIER: float = float(os.getenv("SL_MULTIPLIER", "1.2"))
    TP_MULTIPLIER: float = float(os.getenv("TP_MULTIPLIER", "2.0"))
    MAX_SPREAD_BPS: float = float(os.getenv("MAX_SPREAD_BPS", "5.0"))
    MAX_DRAWDOWN_PCT: float = float(os.getenv("MAX_DRAWDOWN_PCT", "5.0"))
    MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "2"))
    
    LOSS_STREAK_REDUCE_3: float = float(os.getenv("LOSS_STREAK_REDUCE_3", "0.5"))
    LOSS_STREAK_REDUCE_5: float = float(os.getenv("LOSS_STREAK_REDUCE_5", "0.25"))
    
    # File Paths
    JOURNAL_PATH: str = os.getenv("JOURNAL_PATH", "src/data/journal.db")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "src/model/models/lgbm_v1.pkl")
    LOGS_DIR: str = os.getenv("LOGS_DIR", "logs")
    
    # Network & API
    BINANCE_WS_URL: str = os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443/ws")
    BINANCE_REST_URL: str = os.getenv("BINANCE_REST_URL", "https://api.binance.com")
    
    # Telegram Alerter
    TELEGRAM_TOKEN: Optional[str] = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
