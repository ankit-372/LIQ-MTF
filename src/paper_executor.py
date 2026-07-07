import time
from src.shared.event_bus import EventBus

class PaperExecutor:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.active_positions = [] # List of dicts representing open trades (both live and counterfactual)
        
        # Subscribe to AGENT_DECISION_MADE
        EventBus.subscribe("AGENT_DECISION_MADE", self.on_agent_decision_made)
        print("PaperExecutor: Subscribed to 'AGENT_DECISION_MADE'")

    def on_agent_decision_made(self, data: dict):
        """Processes agent decisions, opening live trades or tracking blocked overrides."""
        signal = data.get("signal", "WAIT")
        if signal == "WAIT":
            return
            
        trade_id = data.get("trade_id")
        symbol = data.get("symbol", "BTCUSDT")
        base_size = data.get("base_size", 1000.0)
        final_size = data.get("final_size", 0.0)
        override = data.get("override_triggered", False)
        
        # Retrieve TP/SL targets from the scenario snapshot
        scenario = data.get("scenario", {})
        close_p = float(scenario.get("close", 0.0))
        
        # Use 1.0% defaults if not explicitly present in the decision payload
        tp_threshold = 0.01
        sl_threshold = 0.01
        
        if signal == "BUY":
            tp_price = close_p * (1.0 + tp_threshold)
            sl_price = close_p * (1.0 - sl_threshold)
        else: # SELL
            tp_price = close_p * (1.0 - tp_threshold)
            sl_price = close_p * (1.0 + sl_threshold)
            
        # Determine if this is a live paper trade or a blocked counterfactual simulation
        is_counterfactual = override
        trade_size = base_size if is_counterfactual else final_size
        
        position = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": signal,
            "entry_price": close_p,
            "size": trade_size,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "entry_time": data.get("timestamp", int(time.time())),
            "mfe": 0.0,
            "mae": 0.0,
            "is_counterfactual": is_counterfactual,
            "age": 0 # Count periods to enforce timeout
        }
        
        # Append to active monitoring list
        self.active_positions.append(position)
        
        if not is_counterfactual:
            # Register in the Portfolio tracker for equity checks
            self.portfolio.open_position(symbol, signal, close_p, trade_size)
            print(f"[PaperExecutor] Opened LIVE trade (ID: {trade_id[:8]}, Size: ${trade_size:.2f})")
        else:
            print(f"[PaperExecutor] Opened COUNTERFACTUAL trade for overridden signal (ID: {trade_id[:8]})")

    def update_ticks(self, symbol: str, close: float, high: float, low: float, timestamp: int = None):
        """
        Processes price updates (ticks/bars), checks TP/SL crossings,
        tracks MFE/MAE per tick, and exits positions on trigger.
        """
        current_time = timestamp if timestamp else int(time.time())
        closed_positions = []
        
        for pos in self.active_positions:
            if pos["symbol"] != symbol:
                continue
                
            entry_p = pos["entry_price"]
            side = pos["side"]
            
            # Increments age of the position
            pos["age"] += 1
            
            # Track excursions (MFE / MAE)
            if side == "BUY":
                # MAE: maximum drop below entry price
                drawdown = (entry_p - low) / entry_p
                pos["mae"] = max(pos["mae"], drawdown)
                
                # MFE: maximum runup above entry price
                runup = (high - entry_p) / entry_p
                pos["mfe"] = max(pos["mfe"], runup)
                
                # Check exit conditions
                if low <= pos["sl_price"]:
                    self._close_trade(pos, pos["sl_price"], "STOP_LOSS", current_time)
                    closed_positions.append(pos)
                elif high >= pos["tp_price"]:
                    self._close_trade(pos, pos["tp_price"], "TAKE_PROFIT", current_time)
                    closed_positions.append(pos)
                elif pos["age"] >= 60: # 60-minute timeout vertical barrier
                    self._close_trade(pos, close, "TIMEOUT", current_time)
                    closed_positions.append(pos)
                    
            elif side == "SELL":
                # MAE: maximum rise above entry price
                drawdown = (high - entry_p) / entry_p
                pos["mae"] = max(pos["mae"], drawdown)
                
                # MFE: maximum drop below entry price
                runup = (entry_p - low) / entry_p
                pos["mfe"] = max(pos["mfe"], runup)
                
                # Check exit conditions
                if high >= pos["sl_price"]:
                    self._close_trade(pos, pos["sl_price"], "STOP_LOSS", current_time)
                    closed_positions.append(pos)
                elif low <= pos["tp_price"]:
                    self._close_trade(pos, pos["tp_price"], "TAKE_PROFIT", current_time)
                    closed_positions.append(pos)
                elif pos["age"] >= 60: # 60-minute timeout vertical barrier
                    self._close_trade(pos, close, "TIMEOUT", current_time)
                    closed_positions.append(pos)
                    
        # Remove closed positions from tracking
        for closed in closed_positions:
            self.active_positions.remove(closed)

    def _close_trade(self, pos: dict, exit_price: float, reason: str, exit_time: int):
        """Computes trading metrics and publishes the realized TRADE_CLOSED event."""
        trade_id = pos["trade_id"]
        side = pos["side"]
        entry_price = pos["entry_price"]
        size = pos["size"]
        is_counterfactual = pos["is_counterfactual"]
        
        if side == "BUY":
            realized_pnl = size * (exit_price - entry_price) / entry_price
        else:
            realized_pnl = size * (entry_price - exit_price) / entry_price
            
        if not is_counterfactual:
            # Realized P&L updates portfolio balance
            self.portfolio.close_position(pos["symbol"], exit_price)
            print(f"[PaperExecutor] Realized trade closed (ID: {trade_id[:8]}, Reason: {reason})")
        else:
            print(f"[PaperExecutor] Counterfactual simulation closed (ID: {trade_id[:8]}, Reason: {reason})")
            
        trade_close_data = {
            "trade_id": trade_id,
            "symbol": pos["symbol"],
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_time": exit_time,
            "exit_reason": reason,
            "realized_pnl": realized_pnl,
            "mfe": pos["mfe"],
            "mae": pos["mae"],
            "is_counterfactual": is_counterfactual
        }
        
        # Publish to event bus
        EventBus.publish("TRADE_CLOSED", trade_close_data)
