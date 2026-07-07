import os
import sys
import unittest
import sqlite3
import json
import time
import shutil
import numpy as np
import pandas as pd
import lightgbm as lgb
from types import MappingProxyType

# Add repository root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shared.event_bus import EventBus
from src.shared.feature_registry import FEATURE_COLUMNS
from src.scenario import ScenarioManager
from src.pattern_matcher import PatternMatcher
from src.portfolio import PortfolioTracker
from src.risk_rules import RiskEngine
from src.journal import JournalManager
from src.paper_executor import PaperExecutor
from src.live_executor import LiveExecutor
from src.circuit_breaker import CircuitBreaker

class L4ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Trains and saves a minimal valid LightGBM model for test self-containment."""
        cls.test_model_dir = "tests/mock_models"
        os.makedirs(cls.test_model_dir, exist_ok=True)
        
        # 1. Create dummy dataset of 30 columns (expected by registry)
        np.random.seed(42)
        X = np.random.randn(100, len(FEATURE_COLUMNS))
        y = np.random.randint(0, 3, size=100) # Multiclass label (0, 1, 2)
        
        # 2. Train tiny LightGBM multiclass classifier
        train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_COLUMNS)
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "verbosity": -1,
            "min_data_in_leaf": 5
        }
        booster = lgb.train(params, train_data, num_boost_round=5)
        
        # 3. Save model booster and feature ordering JSON
        booster.save_model(os.path.join(cls.test_model_dir, "model.txt"))
        with open(os.path.join(cls.test_model_dir, "features_order.json"), "w") as f:
            json.dump(FEATURE_COLUMNS, f)
            
        # Write metadata.json for predictor initialization
        with open(os.path.join(cls.test_model_dir, "metadata.json"), "w") as f:
            json.dump({"tp_threshold": 0.01, "sl_threshold": 0.01}, f)

    @classmethod
    def tearDownClass(cls):
        """Cleans up the mock model files."""
        if os.path.exists(cls.test_model_dir):
            shutil.rmtree(cls.test_model_dir)

    def setUp(self):
        from src.predictor import ModelPredictor
        self.predictor = ModelPredictor(model_dir=self.test_model_dir)

    # --- L4 Tests (4 required) ---
    def test_model_booster_loading(self):
        """Test L4.1: Verifies that ModelPredictor initializes and loads the Booster model correctly."""
        self.assertIsNotNone(self.predictor.model)
        self.assertEqual(len(self.predictor.feature_ordering), 45)

    def test_inference_without_error(self):
        """Test L4.2: Verifies that inference runs successfully and generates expected signal structure."""
        # Create a single valid feature row dict
        feature_row = {feat: 0.5 for feat in FEATURE_COLUMNS}
        
        # Call predict on predictor (expects a DataFrame)
        res = self.predictor.predict(pd.DataFrame([feature_row]))
        signal = res.iloc[0]["signal"]
        conf = res.iloc[0]["confidence"]
        self.assertIn(signal, ["WAIT", "BUY", "SELL"])
        self.assertTrue(0.0 <= conf <= 1.0)

    def test_schema_validation_error_on_missing_column(self):
        """Test L4.3: Verifies that passing data missing a required base feature raises ValueError."""
        invalid_row = {feat: 0.5 for feat in FEATURE_COLUMNS if feat != "close"}
        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict(pd.DataFrame([invalid_row]))
        self.assertIn("missing the following required feature columns", str(ctx.exception))

    def test_predictor_handles_reordered_columns(self):
        """Test L4.4: Verifies predictor successfully handles and aligns reordered columns."""
        # Reverse the order of columns in inputs
        reordered_row = {feat: float(i) for i, feat in enumerate(reversed(FEATURE_COLUMNS))}
        
        # Predict should align them internally and return a prediction without errors
        res = self.predictor.predict(pd.DataFrame([reordered_row]))
        signal = res.iloc[0]["signal"]
        self.assertIn(signal, ["WAIT", "BUY", "SELL"])


class L5AgentTests(unittest.TestCase):
    def setUp(self):
        self.test_journal_db = "tests/journal_test.db"
        self.test_matcher_db = "tests/signals_test.db"
        for db in [self.test_journal_db, self.test_matcher_db]:
            if os.path.exists(db):
                os.remove(db)
                
        # Initialize dependencies
        self.portfolio = PortfolioTracker(initial_balance=10000.0)
        self.matcher = PatternMatcher(db_path=self.test_matcher_db)
        self.risk_engine = RiskEngine(base_size=1000.0)
        self.journal = JournalManager(db_path=self.test_journal_db)
        
        # Construct standard passing scenario
        self.scenario = {
            "close": 60000.0,
            "ml_signal": "BUY",
            "ml_confidence": 0.55,
            "market_trend": "BULLISH",
            "volatility_20": 0.005,
            "dist_liq_up_5m": 0.01,
            "dist_liq_below_5m": 0.01,
        }
        # Fake successful pattern matches (>30 matches, positive expectancy)
        self.pattern_matches = {"status": "Success", "win_rate": 0.60, "expectancy": 0.01, "matches": 50}

    def tearDown(self):
        # Unsubscribe database listeners to prevent test interference
        EventBus.clear()
        
        for db in [self.test_journal_db, self.test_matcher_db]:
            if os.path.exists(db):
                try:
                    os.remove(db)
                except Exception:
                    pass

    # --- L5 Agent Tests (10 required) ---
    def test_agent_approves_when_all_checks_pass(self):
        """Test L5.Agent.1: Verifies that trade is approved and full size is projected when all checks pass."""
        decision = self.risk_engine.evaluate_rules(self.scenario, self.portfolio, self.pattern_matches)
        self.assertFalse(decision["override_triggered"])
        self.assertEqual(decision["final_size"], 1000.0)

    def test_agent_blocks_on_drawdown_gt_5(self):
        """Test L5.Agent.2: Verifies that trade is overridden (blocked) if drawdown exceeds 5%."""
        self.portfolio.drawdown_pct = 5.2
        decision = self.risk_engine.evaluate_rules(self.scenario, self.portfolio, self.pattern_matches)
        self.assertTrue(decision["override_triggered"])
        self.assertEqual(decision["final_size"], 0.0)
        self.assertIn("Session drawdown exceeds", decision["override_reason"])

    def test_agent_blocks_on_losses_streak(self):
        """Test L5.Agent.3: Verifies that trade is overridden if loss streak >= 4."""
        self.portfolio.consecutive_losses = 4
        decision = self.risk_engine.evaluate_rules(self.scenario, self.portfolio, self.pattern_matches)
        self.assertTrue(decision["override_triggered"])
        self.assertEqual(decision["final_size"], 0.0)
        self.assertIn("Consecutive loss trades exceed limit", decision["override_reason"])

    def test_agent_blocks_on_max_positions_limit(self):
        """Test L5.Agent.4: Verifies that trade is overridden if open positions >= 3."""
        # Open 3 dummy positions
        self.portfolio.open_position("BTCUSDT", "BUY", 60000.0, 1.0)
        self.portfolio.open_position("ETHUSDT", "BUY", 3000.0, 10.0)
        self.portfolio.open_position("SOLUSDT", "BUY", 150.0, 100.0)
        
        decision = self.risk_engine.evaluate_rules(self.scenario, self.portfolio, self.pattern_matches)
        self.assertTrue(decision["override_triggered"])
        self.assertEqual(decision["final_size"], 0.0)
        self.assertIn("Active position count at maximum", decision["override_reason"])

    def test_agent_blocks_on_wait_signal(self):
        """Test L5.Agent.5: Verifies that trade is overridden if ML signal is WAIT."""
        self.scenario["ml_signal"] = "WAIT"
        decision = self.risk_engine.evaluate_rules(self.scenario, self.portfolio, self.pattern_matches)
        self.assertTrue(decision["override_triggered"])
        self.assertEqual(decision["final_size"], 0.0)

    def test_agent_blocks_on_negative_expectancy(self):
        """Test L5.Agent.6: Verifies that trade is overridden if pattern matcher expectancy is negative."""
        self.pattern_matches["expectancy"] = -0.002
        decision = self.risk_engine.evaluate_rules(self.scenario, self.portfolio, self.pattern_matches)
        self.assertTrue(decision["override_triggered"])
        self.assertEqual(decision["final_size"], 0.0)
        self.assertIn("expectancy is negative", decision["override_reason"])

    def test_agent_reduces_size_on_extreme_volatility(self):
        """Test L5.Agent.7: Verifies size is modified (reduced x0.5) when volatility exceeds 1.2%."""
        self.scenario["volatility_20"] = 0.015
        decision = self.risk_engine.evaluate_rules(self.scenario, self.portfolio, self.pattern_matches)
        self.assertFalse(decision["override_triggered"])
        self.assertEqual(decision["final_size"], 500.0) # 1000.0 * 0.5

    def test_agent_reduces_size_on_counter_trend(self):
        """Test L5.Agent.8: Verifies size is modified (reduced x0.6) on counter-trend signals."""
        self.scenario["market_trend"] = "BEARISH" # Signal is BUY
        decision = self.risk_engine.evaluate_rules(self.scenario, self.portfolio, self.pattern_matches)
        self.assertFalse(decision["override_triggered"])
        self.assertEqual(decision["final_size"], 600.0) # 1000.0 * 0.6

    def test_agent_reduces_size_on_low_confidence(self):
        """Test L5.Agent.9: Verifies size is modified (reduced x0.75) if confidence < 52%."""
        self.scenario["ml_confidence"] = 0.51
        decision = self.risk_engine.evaluate_rules(self.scenario, self.portfolio, self.pattern_matches)
        self.assertFalse(decision["override_triggered"])
        self.assertEqual(decision["final_size"], 750.0) # 1000.0 * 0.75

    def test_agent_compounds_multiple_modifiers(self):
        """Test L5.Agent.10: Verifies multiple modify checks multiply compounding size reductions."""
        self.scenario["volatility_20"] = 0.015  # x0.5
        self.scenario["market_trend"] = "BEARISH" # x0.6
        # Expected compounded multiplier: 0.5 * 0.6 = 0.3
        decision = self.risk_engine.evaluate_rules(self.scenario, self.portfolio, self.pattern_matches)
        self.assertEqual(decision["final_size"], 300.0) # 1000.0 * 0.3

    def test_agent_floors_at_10_percent(self):
        """Test L5.Agent.11: Verifies position sizing multiplier is floored at 10% of base size."""
        self.scenario["volatility_20"] = 0.015    # x0.5
        self.scenario["market_trend"] = "BEARISH"   # x0.6
        self.scenario["ml_confidence"] = 0.51     # x0.75
        self.pattern_matches["status"] = "Insufficient data" # x0.8
        # Compounded: 0.5 * 0.6 * 0.75 * 0.8 = 0.18
        # Floor threshold: 10% of base size ($1,000) = $100. If we add liquidity proximity modify (x0.7), it drops below 10%.
        self.scenario["dist_liq_up_5m"] = 0.002 # triggers proximity modify x0.7
        # Compounded: 0.18 * 0.7 = 0.126. Still above. Let's force lower multipliers
        # Let's check size floor is applied at 10% ($100.00)
        self.risk_engine.base_size = 100.0 # base size = 100. Floored size = 10.
        # Compounded: 100.0 * 0.126 * 0.7 = 8.82. Should floor at 10.0
        decision = self.risk_engine.evaluate_rules(self.scenario, self.portfolio, self.pattern_matches)
        self.assertEqual(decision["final_size"], 10.0)

    def test_journal_writes_decision(self):
        """Test L5.Agent.12: Verifies that JournalManager inserts record to SQLite on decision event."""
        # Setup decision data
        decision_data = {
            "trade_id": "test-decision-uuid-1",
            "timestamp": int(time.time()),
            "symbol": "BTCUSDT",
            "signal": "BUY",
            "base_size": 1000.0,
            "final_size": 1000.0,
            "override_triggered": False,
            "override_reason": "",
            "scenario": self.scenario
        }
        
        # Publish
        EventBus.publish("AGENT_DECISION_MADE", decision_data)
        
        # Read from SQLite
        conn = sqlite3.connect(self.test_journal_db)
        cursor = conn.cursor()
        cursor.execute("SELECT status, base_size FROM trading_journal WHERE trade_id = ?", ("test-decision-uuid-1",))
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "OPEN")
        self.assertEqual(row[1], 1000.0)

    def test_journal_updates_outcome_on_close(self):
        """Test L5.Agent.13: Verifies that JournalManager updates exit fields on TRADE_CLOSED event."""
        # Log initial decision
        decision_data = {
            "trade_id": "test-decision-uuid-2", "timestamp": int(time.time()), "symbol": "BTCUSDT",
            "signal": "BUY", "base_size": 1000.0, "final_size": 1000.0,
            "override_triggered": False, "override_reason": "", "scenario": self.scenario
        }
        EventBus.publish("AGENT_DECISION_MADE", decision_data)
        
        # Publish close
        close_data = {
            "trade_id": "test-decision-uuid-2",
            "symbol": "BTCUSDT",
            "exit_price": 60600.0,
            "exit_time": int(time.time()) + 60,
            "exit_reason": "TAKE_PROFIT",
            "realized_pnl": 10.0,
            "mfe": 0.012,
            "mae": 0.001,
            "is_counterfactual": False
        }
        EventBus.publish("TRADE_CLOSED", close_data)
        
        # Read from SQLite
        conn = sqlite3.connect(self.test_journal_db)
        cursor = conn.cursor()
        cursor.execute("SELECT status, exit_reason, realized_pnl FROM trading_journal WHERE trade_id = ?", ("test-decision-uuid-2",))
        row = cursor.fetchone()
        conn.close()
        
        self.assertEqual(row[0], "CLOSED")
        self.assertEqual(row[1], "TAKE_PROFIT")
        self.assertEqual(row[2], 10.0)


class L5ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.portfolio = PortfolioTracker(initial_balance=10000.0)
        self.paper_executor = PaperExecutor(self.portfolio)
        self.circuit_breaker = CircuitBreaker(self.portfolio, auto_resume_period=2) # 2s resume period
        
        self.trade_closed_received = []
        def on_close(data):
            self.trade_closed_received.append(data)
        EventBus.subscribe("TRADE_CLOSED", on_close)

    def tearDown(self):
        EventBus.clear()

    # --- L5 Executor Tests (8 required) ---
    def test_paper_fill_buy_position(self):
        """Test L5.Exec.1: Verifies that paper executor opens BUY position successfully."""
        decision_data = {
            "trade_id": "exec-uuid-1", "symbol": "BTCUSDT", "signal": "BUY", "base_size": 1000.0,
            "final_size": 1000.0, "override_triggered": False, "scenario": {"close": 60000.0}
        }
        self.paper_executor.on_agent_decision_made(decision_data)
        
        self.assertTrue(self.portfolio.has_open_position("BTCUSDT"))
        self.assertEqual(self.portfolio.open_positions["BTCUSDT"]["side"], "BUY")
        self.assertEqual(self.portfolio.open_positions["BTCUSDT"]["size"], 1000.0)

    def test_paper_fill_sell_position(self):
        """Test L5.Exec.2: Verifies that paper executor opens SELL position successfully."""
        decision_data = {
            "trade_id": "exec-uuid-2", "symbol": "BTCUSDT", "signal": "SELL", "base_size": 1000.0,
            "final_size": 1000.0, "override_triggered": False, "scenario": {"close": 60000.0}
        }
        self.paper_executor.on_agent_decision_made(decision_data)
        
        self.assertTrue(self.portfolio.has_open_position("BTCUSDT"))
        self.assertEqual(self.portfolio.open_positions["BTCUSDT"]["side"], "SELL")

    def test_paper_tracks_mfe_mae_per_tick(self):
        """Test L5.Exec.3: Verifies execution ticks correctly accumulate maximum adverse/favorable excursions (MAE/MFE)."""
        decision_data = {
            "trade_id": "exec-uuid-3", "symbol": "BTCUSDT", "signal": "BUY", "base_size": 1000.0,
            "final_size": 1000.0, "override_triggered": False, "scenario": {"close": 60000.0}
        }
        self.paper_executor.on_agent_decision_made(decision_data)
        
        # Feed ticks: price drops to 59700 (-0.5%), rises to 60300 (+0.5%)
        self.paper_executor.update_ticks("BTCUSDT", close=60000.0, high=60300.0, low=59700.0)
        
        pos = self.paper_executor.active_positions[0]
        self.assertAlmostEqual(pos["mae"], 0.005) # (60000-59700)/60000 = 0.005
        self.assertAlmostEqual(pos["mfe"], 0.005) # (60300-60000)/60000 = 0.005

    def test_paper_sl_detection(self):
        """Test L5.Exec.4: Verifies Stop Loss hit closes trade and publishes exit statistics."""
        decision_data = {
            "trade_id": "exec-uuid-4", "symbol": "BTCUSDT", "signal": "BUY", "base_size": 1000.0,
            "final_size": 1000.0, "override_triggered": False, "scenario": {"close": 60000.0}
        }
        self.paper_executor.on_agent_decision_made(decision_data)
        
        # Entry 60000. SL is 59400 (-1%). Feed low tick of 59300.
        self.paper_executor.update_ticks("BTCUSDT", close=59300.0, high=59500.0, low=59300.0)
        
        self.assertFalse(self.portfolio.has_open_position("BTCUSDT"))
        self.assertEqual(len(self.trade_closed_received), 1)
        self.assertEqual(self.trade_closed_received[0]["exit_reason"], "STOP_LOSS")
        self.assertLess(self.trade_closed_received[0]["realized_pnl"], 0.0)

    def test_paper_tp_detection(self):
        """Test L5.Exec.5: Verifies Take Profit hit closes trade and calculates positive realized PnL."""
        decision_data = {
            "trade_id": "exec-uuid-5", "symbol": "BTCUSDT", "signal": "BUY", "base_size": 1000.0,
            "final_size": 1000.0, "override_triggered": False, "scenario": {"close": 60000.0}
        }
        self.paper_executor.on_agent_decision_made(decision_data)
        
        # Entry 60000. TP is 60600 (+1%). Feed high tick of 60700.
        self.paper_executor.update_ticks("BTCUSDT", close=60700.0, high=60700.0, low=60500.0)
        
        self.assertFalse(self.portfolio.has_open_position("BTCUSDT"))
        self.assertEqual(len(self.trade_closed_received), 1)
        self.assertEqual(self.trade_closed_received[0]["exit_reason"], "TAKE_PROFIT")
        self.assertGreater(self.trade_closed_received[0]["realized_pnl"], 0.0)

    def test_paper_timeout_vertical_barrier(self):
        """Test L5.Exec.6: Verifies position closes automatically under vertical barrier timeout (60m)."""
        decision_data = {
            "trade_id": "exec-uuid-6", "symbol": "BTCUSDT", "signal": "BUY", "base_size": 1000.0,
            "final_size": 1000.0, "override_triggered": False, "scenario": {"close": 60000.0}
        }
        self.paper_executor.on_agent_decision_made(decision_data)
        
        # Feed 60 ticks within limits
        for _ in range(60):
            self.paper_executor.update_ticks("BTCUSDT", close=60000.0, high=60100.0, low=59900.0)
            
        self.assertFalse(self.portfolio.has_open_position("BTCUSDT"))
        self.assertEqual(len(self.trade_closed_received), 1)
        self.assertEqual(self.trade_closed_received[0]["exit_reason"], "TIMEOUT")

    def test_paper_tracks_counterfactuals_on_override(self):
        """Test L5.Exec.7: Verifies paper executor tracks overridden trades virtually without real funds allocation."""
        decision_data = {
            "trade_id": "exec-uuid-7", "symbol": "BTCUSDT", "signal": "BUY", "base_size": 1000.0,
            "final_size": 0.0, "override_triggered": True, "scenario": {"close": 60000.0}
        }
        self.paper_executor.on_agent_decision_made(decision_data)
        
        # Verify it is NOT in live portfolio open positions
        self.assertFalse(self.portfolio.has_open_position("BTCUSDT"))
        
        # Verify it IS in simulated active positions
        self.assertEqual(len(self.paper_executor.active_positions), 1)
        self.assertTrue(self.paper_executor.active_positions[0]["is_counterfactual"])

    def test_breaker_triggers_on_3_sl_hits(self):
        """Test L5.Exec.8: Verifies circuit breaker triggers timed pause when 3 Stop Loss hits occur in 60 minutes."""
        now = int(time.time())
        # Publish 3 SL hits
        EventBus.publish("TRADE_CLOSED", {"trade_id": "dummy-1", "exit_reason": "STOP_LOSS", "exit_time": now, "is_counterfactual": False})
        EventBus.publish("TRADE_CLOSED", {"trade_id": "dummy-2", "exit_reason": "STOP_LOSS", "exit_time": now, "is_counterfactual": False})
        EventBus.publish("TRADE_CLOSED", {"trade_id": "dummy-3", "exit_reason": "STOP_LOSS", "exit_time": now, "is_counterfactual": False})
        
        self.assertTrue(self.circuit_breaker.is_blocked())
        self.assertEqual(self.circuit_breaker.pause_reason, "3 SL hits within 60 minutes")

    def test_breaker_resumes_after_timed_pause(self):
        """Test L5.Exec.9: Verifies timed breaker pause auto-resumes after pause duration elapses."""
        now = int(time.time())
        EventBus.publish("TRADE_CLOSED", {"trade_id": "dummy-a", "exit_reason": "STOP_LOSS", "exit_time": now, "is_counterfactual": False})
        EventBus.publish("TRADE_CLOSED", {"trade_id": "dummy-b", "exit_reason": "STOP_LOSS", "exit_time": now, "is_counterfactual": False})
        EventBus.publish("TRADE_CLOSED", {"trade_id": "dummy-c", "exit_reason": "STOP_LOSS", "exit_time": now, "is_counterfactual": False})
        
        self.assertTrue(self.circuit_breaker.is_blocked())
        
        # Wait for timed pause (we set auto_resume_period=2s)
        time.sleep(2.5)
        self.assertFalse(self.circuit_breaker.is_blocked())

    def test_breaker_drawdown_lock_and_liquidation(self):
        """Test L5.Exec.10: Verifies drawdown > 5% triggers emergency position liquidations and locks trading indefinitely."""
        self.portfolio.open_position("BTCUSDT", "BUY", 60000.0, 20000.0) # $20k size
        self.portfolio.close_position("BTCUSDT", 57000.0) # realized $1,000 loss (10.0% drawdown)
        
        # Trigger breaker check
        self.circuit_breaker.check_drawdown()
        
        self.assertTrue(self.circuit_breaker.is_blocked())
        self.assertTrue(self.circuit_breaker.is_locked)
        self.assertFalse(self.portfolio.has_open_position("BTCUSDT")) # Open positions liquidated

if __name__ == "__main__":
    unittest.main()
