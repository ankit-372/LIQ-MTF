import sys
import os
import time
import sqlite3
import json

# Add repository root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shared.event_bus import EventBus
from src.scenario import ScenarioManager
from src.pattern_matcher import PatternMatcher
from src.portfolio import PortfolioTracker
from src.risk_rules import RiskEngine
from src.journal import JournalManager
from src.paper_executor import PaperExecutor
from src.live_executor import LiveExecutor
from src.circuit_breaker import CircuitBreaker

def run_safety_test():
    print("==================================================")
    print("STARTING EXECUTION & SAFETY PIPELINE INTEGRATION TEST")
    print("==================================================")
    
    # 1. Clean previous test databases
    test_journal_db = "src/data/journal_test.db"
    test_signals_db = "models/signals_history_safety_test.db"
    for db in [test_journal_db, test_signals_db]:
        if os.path.exists(db):
            os.remove(db)
            
    # 2. Initialize all modules
    manager = ScenarioManager()
    matcher = PatternMatcher(db_path=test_signals_db)
    portfolio = PortfolioTracker(initial_balance=10000.0)
    risk_engine = RiskEngine(base_size=1000.0)
    journal = JournalManager(db_path=test_journal_db)
    paper_executor = PaperExecutor(portfolio)
    live_executor = LiveExecutor(use_testnet=True) # Defaults to dry_run as no keys are set
    
    # Initialize CircuitBreaker with 5-second auto-resume period for fast tests
    circuit_breaker = CircuitBreaker(portfolio, auto_resume_period=5)
    
    # Listener to capture events for verification
    closed_trades_logged = []
    def on_trade_closed(data):
        closed_trades_logged.append(data)
    EventBus.subscribe("TRADE_CLOSED", on_trade_closed)
    
    # Setup listener helper to evaluate rules using circuit breaker
    def process_market_cycle(market_tick, ml_pred):
        print(f"\n--- Processing Market Cycle [Signal: {ml_pred['signal']}] ---")
        scenario = manager.package_scenario(market_tick, ml_pred)
        stats = matcher.match_scenario(scenario)
        # Pass circuit_breaker to evaluate_rules
        decision = risk_engine.evaluate_rules(scenario, portfolio, stats, circuit_breaker=circuit_breaker)
        return decision

    # ----------------------------------------------------
    # STEP 1: Successful Trade & MFE/MAE Tick Tracking
    # ----------------------------------------------------
    print("\n>>> STEP 1: Simulating standard BUY signal and position exit...")
    market_tick = {
        "timestamp": int(time.time()),
        "open": 60000.0, "high": 60050.0, "low": 59950.0, "close": 60000.0, "volume": 100.0,
        "quote_volume": 6000000.0, "count": 800, "taker_buy_volume": 55.0,
        "agg_trade_count": 750, "agg_volume": 98.0, "agg_vwap": 60002.0, "buyer_maker_ratio": 0.45,
        "volatility_20": 0.005, "sma_20": 59800.0, "sma_ratio": 1.0033,
        "liquidity_up_5m": 60800.0, "liquidity_below_5m": 59200.0,
        "nearest_liq_5m": 59200.0, "nearest_liq_1h": 59000.0, "nearest_liq_4h": 58000.0, "nearest_liq_1d": 57000.0,
        "dist_liq_up_5m": 0.0133, "dist_liq_below_5m": 0.0133
    }
    ml_pred = {"signal": "BUY", "confidence": 0.55}
    
    decision = process_market_cycle(market_tick, ml_pred)
    assert not decision["override_triggered"], "Error: Expected trade to be approved"
    
    # Entry price is 60000. TP is 60600 (+1%), SL is 59400 (-1%)
    # Let's feed a few ticks and check MAE/MFE accumulation
    # Tick 1: Price goes down slightly (tests MAE)
    paper_executor.update_ticks("BTCUSDT", close=59800.0, high=59900.0, low=59700.0, timestamp=int(time.time()))
    # Tick 2: Price shoots up to hit Take Profit (tests MFE and exit)
    paper_executor.update_ticks("BTCUSDT", close=60650.0, high=60700.0, low=60400.0, timestamp=int(time.time()))
    
    # Check that trade closed and is logged
    assert len(closed_trades_logged) == 1, "Error: Trade should have closed on TP hit"
    trade_outcome = closed_trades_logged[0]
    assert trade_outcome["exit_reason"] == "TAKE_PROFIT", f"Expected TP exit, got {trade_outcome['exit_reason']}"
    assert trade_outcome["realized_pnl"] > 0.0, "Expected positive P&L"
    print(f"  Success: Trade exit reason verified as {trade_outcome['exit_reason']}")
    print(f"  Success: MAE recorded: {trade_outcome['mae']*100:.3f}% | MFE recorded: {trade_outcome['mfe']*100:.3f}%")
    
    # Verify SQLite journal has been updated
    conn = sqlite3.connect(test_journal_db)
    cursor = conn.cursor()
    cursor.execute("SELECT status, exit_reason, realized_pnl FROM trading_journal WHERE trade_id = ?", (trade_outcome["trade_id"],))
    db_record = cursor.fetchone()
    conn.close()
    
    assert db_record is not None, "Error: Trade record not found in journal"
    assert db_record[0] == "CLOSED", f"Expected CLOSED status in DB, got {db_record[0]}"
    assert db_record[1] == "TAKE_PROFIT", f"Expected TAKE_PROFIT in DB, got {db_record[1]}"
    print(f"  Success: SQL Journal status updated to {db_record[0]} with realized PnL of ${db_record[2]:.2f}")

    # ----------------------------------------------------
    # STEP 2: Override & Counterfactual Tracking
    # ----------------------------------------------------
    print("\n>>> STEP 2: Simulating risk override block and counterfactual tracking...")
    # Force consecutive losses in portfolio to trigger override
    portfolio.consecutive_losses = 4
    
    closed_trades_logged.clear()
    decision_2 = process_market_cycle(market_tick, ml_pred)
    assert decision_2["override_triggered"], "Error: Expected override due to consecutive losses"
    assert decision_2["final_size"] == 0.0, "Expected final size to be 0"
    
    # The paper executor should have opened a COUNTERFACTUAL trade
    assert len(paper_executor.active_positions) == 1, "Expected 1 active counterfactual position"
    assert paper_executor.active_positions[0]["is_counterfactual"], "Expected counterfactual flag to be True"
    print("  Success: Counterfactual trade opened successfully.")
    
    # Simulate price drops to hit Stop Loss for the counterfactual position
    # Entry price: 60000. SL price: 59400.
    paper_executor.update_ticks("BTCUSDT", close=59300.0, high=59500.0, low=59300.0, timestamp=int(time.time()))
    
    # Counterfactual trade should have closed
    assert len(closed_trades_logged) == 1, "Error: Counterfactual trade should have closed"
    cf_outcome = closed_trades_logged[0]
    assert cf_outcome["is_counterfactual"], "Expected closed event to flag counterfactual"
    assert cf_outcome["exit_reason"] == "STOP_LOSS", f"Expected SL exit, got {cf_outcome['exit_reason']}"
    print(f"  Success: Counterfactual trade closed on {cf_outcome['exit_reason']} with PnL of ${cf_outcome['realized_pnl']:.2f}")

    # ----------------------------------------------------
    # STEP 3: Circuit Breaker Timed Pause & Auto-Resume
    # ----------------------------------------------------
    print("\n>>> STEP 3: Testing Circuit Breaker 3-SL pause and timed auto-resume...")
    # Reset consecutive losses
    portfolio.consecutive_losses = 0
    
    # Simulate 3 Stop Loss hits (publish manually to event bus)
    now_ts = int(time.time())
    EventBus.publish("TRADE_CLOSED", {"trade_id": "dummy-sl-1", "exit_reason": "STOP_LOSS", "exit_time": now_ts, "is_counterfactual": False})
    EventBus.publish("TRADE_CLOSED", {"trade_id": "dummy-sl-2", "exit_reason": "STOP_LOSS", "exit_time": now_ts, "is_counterfactual": False})
    EventBus.publish("TRADE_CLOSED", {"trade_id": "dummy-sl-3", "exit_reason": "STOP_LOSS", "exit_time": now_ts, "is_counterfactual": False})
    
    # Check that circuit breaker is now blocked
    assert circuit_breaker.is_blocked(), "Error: Circuit breaker should be blocked"
    print(f"  Success: Circuit breaker active. Reason: {circuit_breaker.pause_reason}")
    
    # Try to evaluate rules - should be overridden by circuit breaker block
    decision_3 = process_market_cycle(market_tick, ml_pred)
    assert decision_3["override_triggered"], "Expected override due to active breaker block"
    assert "Circuit Breaker Triggered" in decision_3["override_reason"], f"Expected breaker reason, got: {decision_3['override_reason']}"
    print(f"  Success: Risk Engine correctly overridden by breaker: '{decision_3['override_reason']}'")
    
    # Wait for timed auto-resume period to elapse (we set auto_resume_period = 5s)
    print("  Waiting 6 seconds for timed auto-resume to fire...")
    time.sleep(6)
    
    # Breaker should now be unblocked
    assert not circuit_breaker.is_blocked(), "Error: Circuit breaker should have auto-resumed"
    print("  Success: Circuit breaker successfully auto-resumed after timed pause.")

    # ----------------------------------------------------
    # STEP 4: Circuit Breaker Drawdown Lock & Manual Restart
    # ----------------------------------------------------
    print("\n>>> STEP 4: Testing Circuit Breaker drawdown limit lock and manual reset...")
    # Fake a heavy position in the portfolio tracker and close it at a loss to trigger > 5% drawdown
    portfolio.open_position("BTCUSDT", "BUY", 60000.0, 20000.0)
    # Exiting at 57000 triggers $1,000 loss (~10.0% drawdown)
    portfolio.close_position("BTCUSDT", 57000.0)
    circuit_breaker.check_drawdown()
    
    # Check that circuit breaker is locked
    assert circuit_breaker.is_blocked(), "Error: Breaker should be blocked"
    assert circuit_breaker.is_locked, "Error: Breaker should be locked"
    print(f"  Success: Circuit breaker locked. Reason: {circuit_breaker.pause_reason}")
    
    # Verify that a new signal is blocked
    decision_4 = process_market_cycle(market_tick, ml_pred)
    assert decision_4["override_triggered"], "Expected override due to locked breaker"
    
    # Attempting to wait will NOT auto-resume since it is locked
    print("  Verifying that locked state does not auto-resume on duration...")
    time.sleep(2)
    assert circuit_breaker.is_blocked(), "Error: Breaker should remain blocked indefinitely while locked"
    
    # Perform manual restart with account rebalance to $10,000.00
    circuit_breaker.manual_restart(new_balance=10000.0)
    assert not circuit_breaker.is_blocked(), "Error: Breaker should be active after manual restart"
    assert portfolio.current_balance == 10000.0, "Expected balance reset"
    print("  Success: Manual reset successfully cleared the lock and restored account balance.")
    
    # Clean databases
    for db in [test_journal_db, test_signals_db]:
        if os.path.exists(db):
            try:
                os.remove(db)
            except Exception:
                pass
                
    print("\n==================================================")
    print("ALL SAFETY AND EXECUTION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_safety_test()
