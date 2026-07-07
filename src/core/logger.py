"""
Central Logger for LIQ-MTF.
Subscribes to all Event Bus events and writes JSON lines to daily log files.
"""
import os
import json
from datetime import datetime, timezone
from src.core import event_bus
from src.core.config import Config

class Logger:
    def __init__(self, logs_dir: str = Config.LOGS_DIR):
        self.logs_dir = logs_dir
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            
        # Subscribe to all events on the bus
        event_bus.subscribe_all(self._handle_event)
        
    def _get_log_filepath(self) -> str:
        """Returns the log file path for the current UTC date."""
        current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        return os.path.join(self.logs_dir, f"{current_date}.jsonl")
        
    def _handle_event(self, event_name: str, payload: dict = None):
        """Processes an event and writes it to the log."""
        if payload is None:
            payload = {}
            
        # Ensure payload is a dict, or wrap it
        if not isinstance(payload, dict):
            payload = {"data": payload}
            
        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_name,
            **payload
        }
        
        filepath = self._get_log_filepath()
        
        try:
            with open(filepath, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Failed to write log: {e}")
