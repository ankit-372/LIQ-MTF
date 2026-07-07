import os
import sqlite3
import random
from types import MappingProxyType

class PatternMatcher:
    def __init__(self, db_path: str = "models/signals_history.db"):
        self.db_path = db_path
        # Ensure directories exist
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self._initialize_database()

    def _initialize_database(self):
        """Creates the signals history table if it doesn't exist and seeds it with mock data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS past_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ml_signal TEXT,
                market_trend TEXT,
                volatility_20 REAL,
                dist_liq_up_5m REAL,
                dist_liq_below_5m REAL,
                outcome_pnl REAL
            )
        """)
        conn.commit()
        
        # Check if table is empty. If empty, seed mock signals
        cursor.execute("SELECT COUNT(*) FROM past_signals")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("PatternMatcher: Seeding database with historical signal data...")
            mock_signals = []
            
            # Seed 1,000 mock trade signals
            for _ in range(1000):
                sig = random.choice(["BUY", "SELL"])
                trend = random.choice(["BULLISH", "BEARISH"])
                
                # Volatility centered around 0.005
                vol = max(0.001, random.gauss(0.005, 0.002))
                
                # Distances centered around 0.015 (1.5%)
                dist_up = max(0.001, random.gauss(0.015, 0.005))
                dist_below = max(0.001, random.gauss(0.015, 0.005))
                
                # Setup statistical edge for BUY in BULLISH and SELL in BEARISH
                if (sig == "BUY" and trend == "BULLISH") or (sig == "SELL" and trend == "BEARISH"):
                    pnl = random.gauss(0.012, 0.008) # Positive expectancy (+1.2% avg)
                else:
                    pnl = random.gauss(-0.005, 0.010) # Negative expectancy (-0.5% avg)
                    
                mock_signals.append((sig, trend, vol, dist_up, dist_below, pnl))
                
            cursor.executemany("""
                INSERT INTO past_signals (ml_signal, market_trend, volatility_20, dist_liq_up_5m, dist_liq_below_5m, outcome_pnl)
                VALUES (?, ?, ?, ?, ?, ?)
            """, mock_signals)
            conn.commit()
            
        conn.close()

    def match_scenario(self, scenario: MappingProxyType) -> dict:
        """
        Queries the SQLite database for historical trade signals matching the
        current scenario parameters (Signal, Trend, Volatility, and Liquidity distances).
        
        Returns win rate & expectancy, or 'Insufficient data' if matches < 30.
        """
        ml_signal = scenario.get("ml_signal", "WAIT")
        
        # If model outputs WAIT, there are no trades to match
        if ml_signal == "WAIT":
            return {
                "status": "Success",
                "win_rate": 0.0,
                "expectancy": 0.0,
                "matches": 0
            }
            
        market_trend = scenario.get("market_trend", "BULLISH")
        vol = float(scenario.get("volatility_20", 0.0))
        dist_up = float(scenario.get("dist_liq_up_5m", 0.0))
        dist_below = float(scenario.get("dist_liq_below_5m", 0.0))
        
        # Query matching signals within a +/- 30% range for continuous indicators
        vol_min, vol_max = vol * 0.70, vol * 1.30
        up_min, up_max = dist_up * 0.70, dist_up * 1.30
        below_min, below_max = dist_below * 0.70, dist_below * 1.30
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT outcome_pnl FROM past_signals
            WHERE ml_signal = ?
              AND market_trend = ?
              AND volatility_20 BETWEEN ? AND ?
              AND dist_liq_up_5m BETWEEN ? AND ?
              AND dist_liq_below_5m BETWEEN ? AND ?
        """
        
        cursor.execute(query, (ml_signal, market_trend, vol_min, vol_max, up_min, up_max, below_min, below_max))
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        num_matches = len(results)
        
        # If matches are less than 30, return 'Insufficient data'
        if num_matches < 30:
            return {
                "status": "Insufficient data",
                "win_rate": 0.0,
                "expectancy": 0.0,
                "matches": num_matches
            }
            
        # Calculate win rate and expectancy
        wins = sum(1 for pnl in results if pnl > 0)
        win_rate = float(wins / num_matches)
        expectancy = float(sum(results) / num_matches)
        
        return {
            "status": "Success",
            "win_rate": win_rate,
            "expectancy": expectancy,
            "matches": num_matches
        }
