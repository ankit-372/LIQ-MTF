"""
System Heartbeat for LIQ-MTF.
Monitors components and sends alerts if any fail.
"""
import asyncio
import time
import shutil
import os
from src.core import event_bus
from src.core.alerter import Alerter
from src.core.config import Config

class Heartbeat:
    def __init__(self, alerter: Alerter):
        self.alerter = alerter
        self.last_seen = {
            'ws1': time.time(),
            'ws2': time.time(),
            'ws3': time.time(),
            'ws4': time.time(),
            'feature_engine': time.time()
        }
        
        # Subscribe to internal ping events
        event_bus.subscribe("SYSTEM_PING", self._handle_ping)
        
    def _handle_ping(self, data: dict):
        """Update last_seen timestamp for a given component."""
        component = data.get("component")
        if component in self.last_seen:
            self.last_seen[component] = time.time()
            
    def _check_journal(self) -> bool:
        """Check if SQLite journal is writable."""
        path = Config.JOURNAL_PATH
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            # Just touch the file to ensure it's writable
            with open(path, 'a'):
                pass
            return True
        except Exception:
            return False
            
    def _check_disk_space(self) -> bool:
        """Check if disk has > 1GB free."""
        try:
            total, used, free = shutil.disk_usage("/")
            return free > (1024 * 1024 * 1024)  # 1 GB in bytes
        except Exception:
            return True # Fallback if we can't check
            
    async def run(self):
        """Main loop that runs every 60 seconds."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            
            failures = []
            
            if now - self.last_seen['ws1'] > 5:
                failures.append("WS#1 (aggTrade) delayed > 5s")
            if now - self.last_seen['ws2'] > 10:
                failures.append("WS#2 (kline_5m) delayed > 10s")
            if now - self.last_seen['ws3'] > 5:
                failures.append("WS#3 (bookTicker) delayed > 5s")
            if now - self.last_seen['ws4'] > 120:
                failures.append("WS#4 (kline_1h/4h) delayed > 120s")
            if now - self.last_seen['feature_engine'] > 360:
                failures.append("Feature Engine delayed > 6min")
                
            if not self._check_journal():
                failures.append("Journal (SQLite) is not writable")
                
            if not self._check_disk_space():
                failures.append("Disk space is below 1 GB")
                
            if failures:
                alert_text = "\n".join(failures)
                await self.alerter.send("HEARTBEAT_FAIL", alert_text)
