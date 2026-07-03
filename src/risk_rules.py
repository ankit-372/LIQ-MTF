import time
from src.shared.event_bus import EventBus

class RiskEngine:
    def __init__(self, base_size: float = 1000.0):
        self.base_size = base_size

    def evaluate_rules(self, scenario: dict, portfolio, pattern_matches: dict, symbol: str = "BTCUSDT") -> dict:
        """
        Evaluates 10 risk rules in sequence.
        - OVERRIDE checks immediately abort trading (size = 0).
        - MODIFY checks multiply the trade size.
        - The final size is floored at 10% of base_size (if not overridden).
        - Publishes 'AGENT_DECISION_MADE' event.
        """
        size_multiplier = 1.0
        override_triggered = False
        override_reason = None
        checks_log = []
        
        # Pull parameters from inputs
        close = float(scenario.get("close", 0.0))
        ml_signal = scenario.get("ml_signal", "WAIT")
        ml_confidence = float(scenario.get("ml_confidence", 0.0))
        trend = scenario.get("market_trend", "BULLISH")
        volatility = float(scenario.get("volatility_20", 0.0))
        dist_up = float(scenario.get("dist_liq_up_5m", 0.0))
        dist_below = float(scenario.get("dist_liq_below_5m", 0.0))
        
        # ----------------------------------------------------
        # RULE 1: Max Session Drawdown (OVERRIDE)
        # ----------------------------------------------------
        if portfolio.drawdown_pct >= 5.0:
            override_triggered = True
            override_reason = f"Session drawdown exceeds maximum limit of 5.0% (Current: {portfolio.drawdown_pct:.2f}%)"
            checks_log.append({"rule": 1, "name": "Max Drawdown", "result": "OVERRIDE", "reason": override_reason})
        else:
            checks_log.append({"rule": 1, "name": "Max Drawdown", "result": "PASS"})
            
        # ----------------------------------------------------
        # RULE 2: Max Consecutive Losses (OVERRIDE)
        # ----------------------------------------------------
        if not override_triggered and portfolio.consecutive_losses >= 4:
            override_triggered = True
            override_reason = f"Consecutive loss trades exceed limit of 4 (Current: {portfolio.consecutive_losses})"
            checks_log.append({"rule": 2, "name": "Consecutive Losses Limit", "result": "OVERRIDE", "reason": override_reason})
        else:
            checks_log.append({"rule": 2, "name": "Consecutive Losses Limit", "result": "PASS"})
            
        # ----------------------------------------------------
        # RULE 3: Max Open Position Count (OVERRIDE)
        # ----------------------------------------------------
        if not override_triggered and portfolio.get_open_positions_count() >= 3:
            override_triggered = True
            override_reason = f"Active position count at maximum threshold of 3"
            checks_log.append({"rule": 3, "name": "Open Position Count Limit", "result": "OVERRIDE", "reason": override_reason})
        else:
            checks_log.append({"rule": 3, "name": "Open Position Count Limit", "result": "PASS"})
            
        # ----------------------------------------------------
        # RULE 4: WAIT Signal / No ML Triggers (OVERRIDE)
        # ----------------------------------------------------
        if not override_triggered and ml_signal == "WAIT":
            override_triggered = True
            override_reason = "Model predicted WAIT. No trade signal generated."
            checks_log.append({"rule": 4, "name": "ML Wait Signal", "result": "OVERRIDE", "reason": override_reason})
        else:
            checks_log.append({"rule": 4, "name": "ML Wait Signal", "result": "PASS"})
            
        # ----------------------------------------------------
        # RULE 5: Negative Historical Expectancy (OVERRIDE)
        # ----------------------------------------------------
        # If matches were found but expectancy is negative, block trade
        if not override_triggered and pattern_matches.get("status") == "Success" and pattern_matches.get("expectancy", 0.0) < 0:
            override_triggered = True
            override_reason = f"Historical pattern expectancy is negative ({pattern_matches.get('expectancy'):.4f})"
            checks_log.append({"rule": 5, "name": "Expectancy Check", "result": "OVERRIDE", "reason": override_reason})
        else:
            checks_log.append({"rule": 5, "name": "Expectancy Check", "result": "PASS"})
            
        # ----------------------------------------------------
        # RULE 6: Extreme Volatility Protection (MODIFY)
        # ----------------------------------------------------
        if not override_triggered and volatility > 0.012: # Volatility > 1.2%
            multiplier = 0.5
            size_multiplier *= multiplier
            checks_log.append({"rule": 6, "name": "Extreme Volatility Protection", "result": "MODIFY", "multiplier": multiplier})
        else:
            checks_log.append({"rule": 6, "name": "Extreme Volatility Protection", "result": "PASS"})
            
        # ----------------------------------------------------
        # RULE 7: Trend Alignment Buffer (MODIFY)
        # ----------------------------------------------------
        # Contradicting signals (e.g. BUY in BEARISH trend) gets size reduced
        if not override_triggered:
            contradicts = (ml_signal == "BUY" and trend == "BEARISH") or (ml_signal == "SELL" and trend == "BULLISH")
            if contradicts:
                multiplier = 0.6
                size_multiplier *= multiplier
                checks_log.append({"rule": 7, "name": "Trend Alignment Buffer", "result": "MODIFY", "multiplier": multiplier})
            else:
                checks_log.append({"rule": 7, "name": "Trend Alignment Buffer", "result": "PASS"})
        else:
            checks_log.append({"rule": 7, "name": "Trend Alignment Buffer", "result": "SKIP"})
            
        # ----------------------------------------------------
        # RULE 8: Signal Confidence Level Buffer (MODIFY)
        # ----------------------------------------------------
        if not override_triggered and ml_confidence < 0.52:
            multiplier = 0.75
            size_multiplier *= multiplier
            checks_log.append({"rule": 8, "name": "Low Signal Confidence Buffer", "result": "MODIFY", "multiplier": multiplier})
        else:
            checks_log.append({"rule": 8, "name": "Low Signal Confidence Buffer", "result": "PASS"})
            
        # ----------------------------------------------------
        # RULE 9: Insufficient Pattern Matches Buffer (MODIFY)
        # ----------------------------------------------------
        if not override_triggered and pattern_matches.get("status") == "Insufficient data":
            multiplier = 0.80
            size_multiplier *= multiplier
            checks_log.append({"rule": 9, "name": "Insufficient Pattern Matches", "result": "MODIFY", "multiplier": multiplier})
        else:
            checks_log.append({"rule": 9, "name": "Insufficient Pattern Matches", "result": "PASS"})
            
        # ----------------------------------------------------
        # RULE 10: Proximity to Liquidity Boundaries (MODIFY)
        # ----------------------------------------------------
        # If extremely close (< 0.25% distance) to liquidity limits, reduce size
        if not override_triggered:
            closest_dist = min(dist_up, dist_below)
            if closest_dist < 0.0025:
                multiplier = 0.70
                size_multiplier *= multiplier
                checks_log.append({"rule": 10, "name": "Liquidity Proximity Buffer", "result": "MODIFY", "multiplier": multiplier})
            else:
                checks_log.append({"rule": 10, "name": "Liquidity Proximity Buffer", "result": "PASS"})
        else:
            checks_log.append({"rule": 10, "name": "Liquidity Proximity Buffer", "result": "SKIP"})
            
        # Calculate final trade size
        if override_triggered:
            final_size = 0.0
        else:
            final_size = self.base_size * size_multiplier
            # Floor size at 10% of base trade size
            floor_size = self.base_size * 0.10
            if final_size < floor_size:
                final_size = floor_size
                checks_log.append({"rule": "FLOOR", "name": "Size Floor Constraint", "result": "FLOOR_APPLIED", "value": floor_size})
                
        decision_data = {
            "timestamp": int(time.time()),
            "symbol": symbol,
            "signal": ml_signal,
            "override_triggered": override_triggered,
            "override_reason": override_reason,
            "base_size": self.base_size,
            "final_size": final_size,
            "compounded_multiplier": size_multiplier,
            "checks_evaluated": checks_log
        }
        
        # Publish AGENT_DECISION_MADE event to Central Event Bus
        EventBus.publish("AGENT_DECISION_MADE", decision_data)
        
        return decision_data
