import time
from src.shared.event_bus import EventBus

class CircuitBreaker:
    def __init__(self, portfolio, auto_resume_period: int = 1800):
        self.portfolio = portfolio
        self.auto_resume_period = auto_resume_period # in seconds, default 30 mins
        
        self.is_paused = False
        self.is_locked = False
        self.pause_reason = None
        self.pause_until = 0
        
        self.sl_hits = []           # Timestamps of Stop Loss hits
        self.api_rejections = []    # Timestamps of REST API rejections
        
        self.last_ws_heartbeat = time.time()
        self.last_ws_latency = 0.0
        
        # Subscribe to Event Bus
        EventBus.subscribe("TRADE_CLOSED", self.on_trade_closed)
        EventBus.subscribe("API_REJECTION", self.on_api_rejection)
        EventBus.subscribe("WS_HEARTBEAT", self.on_ws_heartbeat)
        print("CircuitBreaker: Subscribed to 'TRADE_CLOSED', 'API_REJECTION', and 'WS_HEARTBEAT'")

    def on_trade_closed(self, data: dict):
        """Processes closed trades, monitoring SL hits and drawdown limits."""
        exit_reason = data.get("exit_reason", "")
        timestamp = data.get("exit_time", int(time.time()))
        is_counterfactual = data.get("is_counterfactual", False)
        
        if is_counterfactual:
            return # Ignore counterfactuals for breaker checks
            
        if exit_reason == "STOP_LOSS":
            self.register_sl_hit(timestamp)
            
        # Check drawdown from peak via the portfolio tracker
        self.check_drawdown()

    def on_api_rejection(self, data: dict):
        """Processes API rejections to detect connection/execution issues."""
        timestamp = data.get("timestamp", int(time.time()))
        self.register_api_rejection(timestamp)

    def on_ws_heartbeat(self, data: dict):
        """Monitors websocket heartbeats and updates latency state."""
        self.last_ws_heartbeat = time.time()
        self.last_ws_latency = float(data.get("latency_ms", 0.0)) / 1000.0 # convert to seconds
        
        if self.last_ws_latency > 30.0:
            self.trigger_pause(f"WebSocket latency exceeds 30s (Current: {self.last_ws_latency:.2f}s)", 300)

    def register_sl_hit(self, timestamp: int):
        """Tracks SL hits, pausing trading if 3 occur within 60 minutes."""
        self.sl_hits.append(timestamp)
        
        # Filter hits in the last 60 minutes
        cutoff = timestamp - 3600
        self.sl_hits = [t for t in self.sl_hits if t > cutoff]
        
        if len(self.sl_hits) >= 3:
            # Trigger 30-minute timed pause
            self.trigger_pause("3 SL hits within 60 minutes", self.auto_resume_period)

    def register_api_rejection(self, timestamp: int):
        """Tracks API rejections, pausing if 3 occur within 5 minutes."""
        self.api_rejections.append(timestamp)
        
        # Filter rejections in the last 5 minutes (300 seconds)
        cutoff = timestamp - 300
        self.api_rejections = [t for t in self.api_rejections if t > cutoff]
        
        if len(self.api_rejections) >= 3:
            # Trigger 10-minute timed pause
            self.trigger_pause("3 API rejections within 5 minutes", 600)

    def check_drawdown(self):
        """Checks portfolio drawdown. If drawdown exceeds 5%, closes all and locks engine."""
        drawdown = self.portfolio.drawdown_pct
        if drawdown > 5.0:
            self.is_locked = True
            self.is_paused = True
            self.pause_reason = f"Critical session drawdown exceeded 5% Limit (Current: {drawdown:.2f}%)"
            self.pause_until = 0 # Cannot auto-resume
            
            self.liquidate_all_positions()
            
            print(f"[CircuitBreaker CRITICAL] Drawdown reached {drawdown:.2f}%. Liquidated all and LOCKED trading.")

    def liquidate_all_positions(self):
        """Emergency liquidation of all open positions in portfolio."""
        open_symbols = list(self.portfolio.open_positions.keys())
        for symbol in open_symbols:
            pos = self.portfolio.open_positions[symbol]
            entry_p = pos["entry_price"]
            # Close at current market price (simulate close at entry price to represent emergency dump)
            self.portfolio.close_position(symbol, entry_p)
            print(f"[CircuitBreaker Emergency] Closed position on {symbol} to prevent further drawdown.")

    def trigger_pause(self, reason: str, duration: int):
        """Triggers a temporary pause with timed auto-resume."""
        if self.is_locked:
            return # Locked state takes absolute priority
            
        self.is_paused = True
        self.pause_reason = reason
        self.pause_until = time.time() + duration
        print(f"[CircuitBreaker Timed Pause] Trading paused for {duration}s. Reason: {reason}")

    def manual_restart(self, new_balance: float = None):
        """Restores locked circuit breaker and resets metrics."""
        self.is_locked = False
        self.is_paused = False
        self.pause_reason = None
        self.pause_until = 0
        self.sl_hits.clear()
        self.api_rejections.clear()
        self.last_ws_heartbeat = time.time()
        
        if new_balance:
            self.portfolio.initial_balance = new_balance
            self.portfolio.current_balance = new_balance
            self.portfolio.peak_balance = new_balance
            self.portfolio.drawdown_pct = 0.0
            self.portfolio.consecutive_losses = 0
            
        print("[CircuitBreaker] Manual system restart executed successfully. Trading resumed.")

    def is_blocked(self) -> bool:
        """
        Returns True if trading is blocked due to active pause/lock.
        Evaluates timed auto-resume and WebSocket latency timeouts.
        """
        now = time.time()
        
        # Check if auto-resume timed out
        if self.is_paused and not self.is_locked and self.pause_until > 0:
            if now >= self.pause_until:
                self.is_paused = False
                self.pause_reason = None
                self.pause_until = 0
                print("[CircuitBreaker Auto-Resume] Pause duration elapsed. Autoresumed trading.")
                
        # Check WS Heartbeat age (if no messages received for > 30s, trigger pause)
        if now - self.last_ws_heartbeat > 30.0:
            if not self.is_paused:
                self.is_paused = True
                self.pause_reason = f"WebSocket heartbeat timeout > 30s (Last seen: {now - self.last_ws_heartbeat:.1f}s ago)"
                self.pause_until = 0 # Must wait for a heartbeat event to resume
                print(f"[CircuitBreaker Warning] {self.pause_reason}")
                
        return self.is_paused or self.is_locked
