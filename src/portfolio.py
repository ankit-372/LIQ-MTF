class PortfolioTracker:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        self.session_pnl = 0.0          # Cumulative realized dollar P&L
        self.drawdown_pct = 0.0         # Current session drawdown percentage from peak
        self.consecutive_losses = 0     # Current streak of consecutive loss trades
        
        self.open_positions = {}        # Active positions dict: {symbol: position_details}
        self.closed_trades = []         # Realized trade logs

    def get_open_positions_count(self) -> int:
        """Returns the number of active open positions."""
        return len(self.open_positions)

    def has_open_position(self, symbol: str) -> bool:
        """Returns True if there is an active position for the given symbol."""
        return symbol in self.open_positions

    def open_position(self, symbol: str, side: str, entry_price: float, size: float):
        """
        Opens a new position and registers it in the tracker.
        'side' must be either 'BUY' or 'SELL'.
        """
        if self.has_open_position(symbol):
            print(f"[Portfolio Warning] Position already exists for {symbol}. Ignoring open request.")
            return
            
        self.open_positions[symbol] = {
            "side": side.upper(),
            "entry_price": float(entry_price),
            "size": float(size)
        }
        print(f"[Portfolio] Opened {side.upper()} position on {symbol} at {entry_price} (size={size})")

    def close_position(self, symbol: str, exit_price: float) -> dict:
        """
        Closes an active position, calculates realized P&L, updates drawdown
        metrics, and logs consecutive loss streak.
        """
        if not self.has_open_position(symbol):
            print(f"[Portfolio Warning] No open position found for {symbol} to close.")
            return {}
            
        pos = self.open_positions.pop(symbol)
        side = pos["side"]
        entry_price = pos["entry_price"]
        size = pos["size"]
        
        # Calculate raw dollar P&L
        if side == "BUY":
            trade_pnl = size * (exit_price - entry_price)
        else:
            trade_pnl = size * (entry_price - exit_price)
            
        # Update balance and realize P&L
        self.session_pnl += trade_pnl
        self.current_balance += trade_pnl
        
        # Track consecutive loss streak
        if trade_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # Update peak balance and drawdown from peak
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            
        # Drawdown = (Peak - Current) / Peak
        self.drawdown_pct = float((self.peak_balance - self.current_balance) / self.peak_balance * 100.0)
        
        trade_record = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl": trade_pnl,
            "pnl_pct": float(trade_pnl / (entry_price * size) * 100.0) if entry_price * size > 0 else 0.0
        }
        self.closed_trades.append(trade_record)
        
        print(f"[Portfolio] Closed {side} position on {symbol} at {exit_price}. P&L: ${trade_pnl:.2f} ({trade_record['pnl_pct']:.2f}%)")
        print(f"[Portfolio] Session Status - Balance: ${self.current_balance:.2f} | Peak: ${self.peak_balance:.2f} | Drawdown: {self.drawdown_pct:.2f}% | Loss Streak: {self.consecutive_losses}")
        
        return trade_record
