"""
Telegram Alerter for LIQ-MTF.
Sends critical notifications via Telegram Bot API.
"""
import asyncio
import urllib.request
import urllib.parse
import json
from src.core.config import Config

class Alerter:
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or Config.TELEGRAM_TOKEN
        self.chat_id = chat_id or Config.TELEGRAM_CHAT_ID
        
    def _send_sync(self, message: str) -> bool:
        """Synchronously sends a message to Telegram."""
        if not self.token or not self.chat_id:
            # Skip if not configured
            return False
            
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = json.dumps({
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=5.0) as response:
                return response.status == 200
        except Exception as e:
            print(f"Failed to send Telegram alert: {e}")
            return False

    async def send(self, event_type: str, details: str = "") -> bool:
        """
        Asynchronously sends an alert.
        Supported event types: TRADE_OPENED, TRADE_CLOSED, CIRCUIT_BREAKER, HEARTBEAT_FAIL, ORDER_REJECTED
        """
        message = f"<b>{event_type}</b>\n{details}"
        # Run the synchronous network call in a thread pool to avoid blocking asyncio loop
        return await asyncio.to_thread(self._send_sync, message)
