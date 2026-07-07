import os
import sqlite3
import json
import uuid
import time
from src.shared.event_bus import EventBus

class JournalManager:
    def __init__(self, db_path: str = "src/data/journal.db"):
        self.db_path = db_path
        
        # Ensure directories exist
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self._initialize_database()
        
        # Subscribe to Event Bus
        EventBus.subscribe("AGENT_DECISION_MADE", self.on_agent_decision_made)
        EventBus.subscribe("TRADE_CLOSED", self.on_trade_closed)
        print("JournalManager: Subscribed to 'AGENT_DECISION_MADE' and 'TRADE_CLOSED'")

    def _initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trading journal table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_journal (
                trade_id TEXT PRIMARY KEY,
                timestamp INTEGER,
                symbol TEXT,
                signal TEXT,
                base_size REAL,
                final_size REAL,
                override_triggered INTEGER,
                override_reason TEXT,
                scenario_snapshot TEXT,
                status TEXT,
                exit_price REAL,
                exit_time INTEGER,
                exit_reason TEXT,
                realized_pnl REAL,
                mfe REAL,
                mae REAL
            )
        """)
        conn.commit()
        conn.close()

    def on_agent_decision_made(self, data: dict):
        """Logs the signal and decision snapshot to SQLite database."""
        # Check if we should track this cycle
        signal = data.get("signal", "WAIT")
        if signal == "WAIT":
            return # Don't journal non-signals
            
        override = 1 if data.get("override_triggered", False) else 0
        trade_id = data.get("trade_id")
        if not trade_id:
            trade_id = str(uuid.uuid4())
            data["trade_id"] = trade_id
        
        # Convert scenario to JSON string
        scenario = data.get("scenario", {})
        # Convert MappingProxyType to regular dict for serialization if needed
        if hasattr(scenario, "copy"):
            scenario_dict = dict(scenario)
        else:
            scenario_dict = {}
            
        scenario_json = json.dumps(scenario_dict)
        
        # Decide initial status
        if override:
            status = "OVERRIDDEN"
        else:
            status = "OPEN"
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trading_journal (
                trade_id, timestamp, symbol, signal, base_size, final_size,
                override_triggered, override_reason, scenario_snapshot, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id,
            data.get("timestamp", int(time.time())),
            data.get("symbol", "BTCUSDT"),
            signal,
            data.get("base_size", 0.0),
            data.get("final_size", 0.0),
            override,
            data.get("override_reason", ""),
            scenario_json,
            status
        ))
        conn.commit()
        conn.close()
        print(f"[Journal] Logged decision to database (ID: {trade_id[:8]}, Status: {status})")

    def on_trade_closed(self, data: dict):
        """Updates trade outcome fields when trade closes."""
        trade_id = data.get("trade_id")
        if not trade_id:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update P&L and metrics
        cursor.execute("""
            UPDATE trading_journal
            SET status = ?,
                exit_price = ?,
                exit_time = ?,
                exit_reason = ?,
                realized_pnl = ?,
                mfe = ?,
                mae = ?
            WHERE trade_id = ?
        """, (
            "CLOSED",
            data.get("exit_price", 0.0),
            data.get("exit_time", 0),
            data.get("exit_reason", ""),
            data.get("realized_pnl", 0.0),
            data.get("mfe", 0.0),
            data.get("mae", 0.0),
            trade_id
        ))
        conn.commit()
        conn.close()
        pnl = data.get("realized_pnl")
        pnl_val = float(pnl) if pnl is not None else 0.0
        print(f"[Journal] Updated trade outcome in database (ID: {trade_id[:8]}, PnL: ${pnl_val:.2f})")
