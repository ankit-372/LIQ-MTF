import os
import time
import hmac
import hashlib
import urllib.parse
import threading
import requests
from src.shared.event_bus import EventBus

class LiveExecutor:
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 tg_token: str = None, tg_chat_id: str = None, use_testnet: bool = True):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        self.tg_token = tg_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.tg_chat_id = tg_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        self.use_testnet = use_testnet
        if self.use_testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
            
        # If API keys are missing, execute in Mock/Dry-Run mode for safety
        self.dry_run = not (self.api_key and self.api_secret)
        if self.dry_run:
            print("[LiveExecutor Warning] API keys missing. Operating in Dry-Run/Mock mode.")
            
        # Subscribe to AGENT_DECISION_MADE
        EventBus.subscribe("AGENT_DECISION_MADE", self.on_agent_decision_made)
        print("LiveExecutor: Subscribed to 'AGENT_DECISION_MADE'")

    def on_agent_decision_made(self, data: dict):
        """Monitors decisions. If trade is approved, spawns executor thread."""
        signal = data.get("signal", "WAIT")
        if signal == "WAIT":
            return
            
        override = data.get("override_triggered", False)
        final_size = data.get("final_size", 0.0)
        
        # Abort if override triggered or final size is 0
        if override or final_size <= 0.0:
            return
            
        trade_id = data.get("trade_id")
        symbol = data.get("symbol", "BTCUSDT")
        
        scenario = data.get("scenario", {})
        close_p = float(scenario.get("close", 0.0))
        
        # Default 1% TP / 1% SL thresholds
        tp_threshold = 0.01
        sl_threshold = 0.01
        
        if signal == "BUY":
            tp_price = close_p * (1.0 + tp_threshold)
            sl_price = close_p * (1.0 - sl_threshold)
        else:
            tp_price = close_p * (1.0 - tp_threshold)
            sl_price = close_p * (1.0 + sl_threshold)
            
        # Convert USD size to BTC contract size before placing order
        btc_quantity = final_size / close_p
        # Format to Binance decimals (BTC Futures accepts up to 3 decimals)
        formatted_qty = round(btc_quantity, 3) 
        if formatted_qty < 0.001:
            print(f"[LiveExecutor Error] Formatted quantity {formatted_qty} is below Binance minimum of 0.001 BTC")
            return
            
        # Run execution in a separate thread to avoid blocking event loop
        thread = threading.Thread(
            target=self.execute_trade_flow,
            args=(trade_id, symbol, signal, formatted_qty, close_p, tp_price, sl_price)
        )
        thread.daemon = True
        thread.start()

    def send_telegram_alert(self, message: str):
        """Sends an emergency alert message to Telegram bot."""
        print(f"[Telegram Alert] {message}")
        if self.dry_run or not self.tg_token or not self.tg_chat_id:
            return
            
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        payload = {"chat_id": self.tg_chat_id, "text": message, "parse_mode": "Markdown"}
        try:
            res = requests.post(url, json=payload, timeout=5)
            res.raise_for_status()
        except Exception as e:
            print(f"[Telegram Error] Failed to send telegram message: {e}")

    def _sign_payload(self, params: dict) -> str:
        """Generates HMAC-SHA256 signature for Binance request payload."""
        query_string = urllib.parse.urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _send_signed_request(self, method: str, path: str, params: dict) -> dict:
        """Sends an authenticated request to Binance Futures REST API."""
        if self.dry_run:
            # Mock successful response
            return {"orderId": 99999999, "status": "FILLED"}
            
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign_payload(params)
        
        url = self.base_url + path
        headers = {"X-MBX-APIKEY": self.api_key}
        
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, data=params, timeout=10)
        else:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
        response.raise_for_status()
        return response.json()

    def execute_trade_flow(self, trade_id: str, symbol: str, side: str, 
                           size: float, entry_price: float, tp_price: float, sl_price: float):
        """Runs full REST order flow: entry order -> bracket orders -> polling fill loop."""
        print(f"[LiveExecutor] Initiating order flow for {side} {symbol}...")
        
        # Step 1: Entry Order
        entry_order = self._place_order_with_retry(
            symbol=symbol,
            side=side,
            order_type="MARKET",
            quantity=size,
            desc=f"Entry {side} Order"
        )
        
        if not entry_order:
            self.send_telegram_alert(f"⚠️ *CRITICAL*: Entry Order failed for trade {trade_id[:8]} on {symbol}. Execution aborted.")
            return
            
        print(f"[LiveExecutor] Entry Order Filled! Order ID: {entry_order.get('orderId')}")
        
        # Step 2: Bracket Orders (STOP_MARKET & TAKE_PROFIT_MARKET)
        opposite_side = "SELL" if side == "BUY" else "BUY"
        
        # Place Stop Loss
        sl_order = self._place_order_with_retry(
            symbol=symbol,
            side=opposite_side,
            order_type="STOP_MARKET",
            quantity=size,
            stop_price=sl_price,
            close_position=True,
            desc="Bracket Stop Loss Order"
        )
        
        # Place Take Profit
        tp_order = self._place_order_with_retry(
            symbol=symbol,
            side=opposite_side,
            order_type="TAKE_PROFIT_MARKET",
            quantity=size,
            stop_price=tp_price,
            close_position=True,
            desc="Bracket Take Profit Order"
        )
        
        if not sl_order or not tp_order:
            self.send_telegram_alert(
                f"⚠️ *WARNING*: Bracket SL/TP placement failed for trade {trade_id[:8]} on {symbol}. "
                f"SL: {'SUCCESS' if sl_order else 'FAIL'}, TP: {'SUCCESS' if tp_order else 'FAIL'}. Manual intervention required!"
            )
            
        # Step 3: Poll Fills (Every 5s)
        # For dry-run/mock, we simulate exit after a short loop
        if self.dry_run:
            print("[LiveExecutor] Simulating fill polling loop in dry-run...")
            time.sleep(10)
            print("[LiveExecutor] Mock trade exit triggered.")
            EventBus.publish("TRADE_CLOSED", {
                "trade_id": trade_id, "symbol": symbol, "side": side,
                "entry_price": entry_price, "exit_price": tp_price, "exit_time": int(time.time()),
                "exit_reason": "TAKE_PROFIT", "realized_pnl": size * (tp_price - entry_price) if side == "BUY" else size * (entry_price - tp_price),
                "mfe": 0.015, "mae": 0.002, "is_counterfactual": False
            })
            return
            
        # Active polling loop for live trades
        sl_id = sl_order.get("orderId") if sl_order else None
        tp_id = tp_order.get("orderId") if tp_order else None
        
        print("[LiveExecutor] Starting live fill polling loop (every 5 seconds)...")
        while True:
            time.sleep(5)
            try:
                # Check status of SL and TP orders
                sl_filled = self._check_order_filled(symbol, sl_id) if sl_id else False
                tp_filled = self._check_order_filled(symbol, tp_id) if tp_id else False
                
                if sl_filled or tp_filled:
                    exit_reason = "STOP_LOSS" if sl_filled else "TAKE_PROFIT"
                    exit_price = sl_price if sl_filled else tp_price
                    print(f"[LiveExecutor] Bracket filled! Reason: {exit_reason}")
                    
                    # Cancel the other order (Clean up remaining bracket leg)
                    other_id = tp_id if sl_filled else sl_id
                    if other_id:
                        self._cancel_order(symbol, other_id)
                        
                    # Calculate realized P&L
                    if side == "BUY":
                        realized_pnl = size * (exit_price - entry_price)
                    else:
                        realized_pnl = size * (entry_price - exit_price)
                        
                    # Publish TRADE_CLOSED
                    EventBus.publish("TRADE_CLOSED", {
                        "trade_id": trade_id,
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "exit_time": int(time.time()),
                        "exit_reason": exit_reason,
                        "realized_pnl": realized_pnl,
                        "mfe": 0.0, # Live executor depends on exchange details, MFE/MAE is 0 unless calculated
                        "mae": 0.0,
                        "is_counterfactual": False
                    })
                    break
            except Exception as e:
                print(f"[LiveExecutor Error] Exception in polling loop: {e}")

    def _place_order_with_retry(self, symbol: str, side: str, order_type: str, 
                                quantity: float, stop_price: float = None, 
                                close_position: bool = False, desc: str = "") -> dict:
        """Attempts to place an order, retrying once on failure."""
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if stop_price:
            params["stopPrice"] = f"{stop_price:.2f}"
        if close_position:
            params["closePosition"] = "true"
            
        path = "/fapi/v1/order"
        
        # Try once
        try:
            return self._send_signed_request("POST", path, params)
        except Exception as e:
            print(f"[LiveExecutor Warning] Failed placing {desc} ({e}). Retrying once...")
            time.sleep(1)
            # Retry once
            try:
                return self._send_signed_request("POST", path, params)
            except Exception as e2:
                print(f"[LiveExecutor Error] Retry failed for {desc}: {e2}")
                return None

    def _check_order_filled(self, symbol: str, order_id: int) -> bool:
        """Polls Binance for order status."""
        path = "/fapi/v1/order"
        params = {"symbol": symbol, "orderId": str(order_id)}
        try:
            res = self._send_signed_request("GET", path, params)
            return res.get("status") == "FILLED"
        except Exception as e:
            print(f"[LiveExecutor Error] Failed checking order status: {e}")
            return False

    def _cancel_order(self, symbol: str, order_id: int):
        """Cancels open order."""
        path = "/fapi/v1/order"
        params = {"symbol": symbol, "orderId": str(order_id)}
        try:
            self._send_signed_request("DELETE", path, params)
            print(f"[LiveExecutor] Cancelled order {order_id}")
        except Exception as e:
            print(f"[LiveExecutor Error] Failed to cancel order {order_id}: {e}")
