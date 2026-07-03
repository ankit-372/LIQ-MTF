import sys
import os
import time

# Add repository root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shared.event_bus import EventBus
from src.scenario import ScenarioManager
from src.pattern_matcher import PatternMatcher
from src.portfolio import PortfolioTracker
from src.risk_rules import RiskEngine

def run_pipeline_simulation():
    print("==================================================")
    print("STARTING AGENT RISK PIPELINE SIMULATION")
    print("==================================================")
    
    # 1. Initialize all modules
    manager = ScenarioManager()
    matcher = PatternMatcher(db_path="models/signals_history_test.db")
    portfolio = PortfolioTracker(initial_balance=10000.0)
    risk_engine = RiskEngine(base_size=1000.0)
    
    # Track final decision output
    final_decisions = []
    
    def on_decision_made(data):
        print("\n>>> Event 'AGENT_DECISION_MADE' Received!")
        print(f"  Symbol: {data['symbol']}")
        print(f"  Signal: {data['signal']}")
        print(f"  Override Status: {data['override_triggered']} (Reason: {data['override_reason']})")
        print(f"  Base Size: ${data['base_size']:.2f}")
        print(f"  Final Position Size: ${data['final_size']:.2f}")
        print(f"  Compounded Size Multiplier: {data['compounded_multiplier']:.4f}")
        print("  Detailed Checks Log:")
        for check in data['checks_evaluated']:
            res = check.get("result")
            mult = f" (Mult: {check.get('multiplier')})" if "multiplier" in check else ""
            reason = f" - Reason: {check.get('reason')}" if "reason" in check else ""
            print(f"    - Rule {check['rule']} [{check['name']}]: {res}{mult}{reason}")
        final_decisions.append(data)
        
    def on_scenario_created(data):
        print("\n>>> Event 'SCENARIO_CREATED' Received!")
        scenario = data["scenario"]
        print(f"  Packaged {len(scenario)} frozen fields.")
        print(f"  ML Signal: {scenario['ml_signal']} (Confidence: {scenario['ml_confidence']:.2f})")
        print(f"  Volatility: {scenario['volatility_20']:.4f} | Trend: {scenario['market_trend']}")
        
        # Ingest into PatternMatcher
        print("  Executing pattern matcher SQLite query...")
        stats = matcher.match_scenario(scenario)
        print(f"  Pattern matches found: {stats['matches']} | Status: {stats['status']}")
        if stats['status'] == "Success":
            print(f"  Historical Win Rate: {stats['win_rate']*100:.2f}% | Expectancy: {stats['expectancy']:.4f}")
            
        # Ingest into RiskEngine
        print("  Evaluating Risk Rules...")
        risk_engine.evaluate_rules(scenario, portfolio, stats)

    # Subscribe to target events
    EventBus.subscribe("SCENARIO_CREATED", on_scenario_created)
    EventBus.subscribe("AGENT_DECISION_MADE", on_decision_made)
    
    # ----------------------------------------------------
    # TEST CASE 1: Standard BULLISH BUY Signal (Passes/Modifies)
    # ----------------------------------------------------
    print("\n--- TEST CASE 1: Standard BUY signal in BULLISH trend ---")
    market_tick = {
        "timestamp": int(time.time()),
        "open": 58500.0, "high": 58650.0, "low": 58450.0, "close": 58600.0, "volume": 120.0,
        "quote_volume": 10200000.0, "count": 1050, "taker_buy_volume": 65.0,
        "agg_trade_count": 1000, "agg_volume": 118.0, "agg_vwap": 58595.0, "buyer_maker_ratio": 0.44,
        "volatility_20": 0.006, # Moderate volatility (No volatility modify)
        "sma_20": 58400.0, # Close > SMA (BULLISH trend)
        "sma_ratio": 1.0034,
        "liquidity_up_5m": 59200.0, "liquidity_below_5m": 58100.0,
        "nearest_liq_5m": 58100.0, "nearest_liq_1h": 58000.0, "nearest_liq_4h": 57500.0, "nearest_liq_1d": 57000.0,
        "dist_liq_up_5m": 0.0102, "dist_liq_below_5m": 0.0085 # Distance > 0.25% (No close-boundary modify)
    }
    ml_prediction = {
        "signal": "BUY",
        "confidence": 0.54 # Confidence > 52% (No confidence modify)
    }
    
    EventBus.publish("ML_SIGNAL_GENERATED", {"market_tick": market_tick, "ml_prediction": ml_prediction})
    
    # ----------------------------------------------------
    # TEST CASE 2: Counter-trend BUY Signal with Low Confidence (Compounding Modifies)
    # ----------------------------------------------------
    print("\n--- TEST CASE 2: Counter-trend BUY (BEARISH trend) & Low Confidence (Compounding Modifies) ---")
    market_tick_2 = market_tick.copy()
    market_tick_2["sma_20"] = 58800.0 # Close (58600) < SMA (58800) -> BEARISH trend (Triggers Trend modify)
    market_tick_2["volatility_20"] = 0.015 # Extremely high volatility (Triggers Volatility modify)
    
    ml_prediction_2 = {
        "signal": "BUY",
        "confidence": 0.51 # Confidence < 52% (Triggers Confidence modify)
    }
    
    # Compounding multipliers: 0.6 (trend) * 0.5 (volatility) * 0.75 (confidence) = 0.225
    # Expected size = 1000 * 0.225 = $225.0
    EventBus.publish("ML_SIGNAL_GENERATED", {"market_tick": market_tick_2, "ml_prediction": ml_prediction_2})
    
    # ----------------------------------------------------
    # TEST CASE 3: Heavy Drawdown (Triggers OVERRIDE)
    # ----------------------------------------------------
    print("\n--- TEST CASE 3: Session Drawdown Limit Exceeded (Triggers OVERRIDE) ---")
    portfolio.open_position("BTCUSDT", "BUY", 58600.0, 1.0)
    # Force close at a heavy loss to trigger drawdown
    portfolio.close_position("BTCUSDT", 55000.0) # realize -$3,600.0 loss (~36.0% drawdown)
    
    # Attempt to process trade signals
    EventBus.publish("ML_SIGNAL_GENERATED", {"market_tick": market_tick, "ml_prediction": ml_prediction})
    
    # Clean up test database
    if os.path.exists("models/signals_history_test.db"):
        os.remove("models/signals_history_test.db")
        
    print("\n==================================================")
    print("ALL PIPELINE TESTS EXECUTED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline_simulation()
