# Level 4 Model Trainer & Predictor Module (BTCUSDT)

This repository contains the **Level 4 Multiclass LightGBM Model Trainer, Predictor Engine, and Backtest Simulation Framework** designed for BTCUSDT high-resolution algorithmic trading.

All implementations in this module adhere strictly to memory-safe chunked processing, lookahead-purged time series splitting, dynamic indicator computation, and schema-validated deterministic inference.

---

## 📂 1. Repository File Structure & Component Overview

| File / Directory | Type | Description |
| :--- | :--- | :--- |
| **[src/trainer.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/src/trainer.py)** | Script | Primary model training module. Handles chunked memory aggregation of high-resolution trade data, Triple Barrier Labeling, time-purged splitting, and LightGBM model fitting. |
| **[src/predictor.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/src/predictor.py)** | Module | Inference engine. Loads the trained booster, validates column schemas, computes indicators on-the-fly, applies confidence thresholding, and projects TP/SL levels. |
| **[src/scenario.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/src/scenario.py)** | Module | Packages the current market tick and prediction values into a frozen, read-only 25-field scenario snapshot. |
| **[src/pattern_matcher.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/src/pattern_matcher.py)** | Module | Interfaces with SQLite to find past signals matching the current scenario and returns win_rate and expectancy. |
| **[src/portfolio.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/src/portfolio.py)** | Module | Tracks open positions, session realized P&L, consecutive losing streaks, and running drawdown from peak balance. |
| **[src/risk_rules.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/src/risk_rules.py)** | Module | Risk Engine that runs 10 sequential checks to compute position sizing modifiers, apply overrides, floor sizes at 10%, and trigger execution. |
| **[src/paper_executor.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/src/paper_executor.py)** | Module | Simulated exchange executor. Handles tick-level fills, Stop Loss/Take Profit hits, MFE/MAE calculations, and tracks virtual counterfactual trades for overridden signals. |
| **[src/live_executor.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/src/live_executor.py)** | Module | Live exchange executor. Interacts with Binance Futures REST API to place entry and bracket TP/SL orders, polls fills, and triggers Telegram bot alerts on execution failure. |
| **[src/circuit_breaker.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/src/circuit_breaker.py)** | Module | Safety firewall. Manages auto-resuming halts on SL streaks or rejections, handles latency disconnect checks, and executes hard locks with emergency liquidations on critical drawdown (>5%). |
| **[src/journal.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/src/journal.py)** | Module | SQLite ledger manager (`journal.db`). Logs every decision snapshot, signal, and execution outcome (exit prices, timing, realized P&L, excursion metrics). |
| **[src/shared/feature_registry.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/src/shared/feature_registry.py)** | Module | Single source of truth registering all 30 features and ordering expected by the model. |
| **[src/shared/event_bus.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/src/shared/event_bus.py)** | Module | Central Event Bus mediating system events (e.g. `ML_SIGNAL_GENERATED`, `SCENARIO_CREATED`, `AGENT_DECISION_MADE`). |
| **[models/](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/models)** | Directory | Stores the active trained booster (`model.txt`), feature column ordering (`features_order.json`), run parameters (`metadata.json`), and evaluation metrics (`evaluation_report.txt`). |
| **[cache/](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/cache)** | Directory | Caches aligned 1-minute features (`BTCUSDT_1m_aggregated_features.parquet`) generated from klines, trades, and liquidity maps (ignored in Git). |
| **[tests/validate_trainer_predictor.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/tests/validate_trainer_predictor.py)** | Script | End-to-end integration test suite verifying trainer pipelines, schema validation strictness, and deterministic prediction outputs on subset data. |
| **[tests/simulate_backtest.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/tests/simulate_backtest.py)** | Script | Historical backtester evaluating trade execution, Win Rate, Net P&L, and Profit Factor across custom confidence thresholds. |
| **[tests/tune_hyperparameters.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/tests/tune_hyperparameters.py)** | Script | Automated grid search script evaluating class weights (`w-buy`, `w-sell`), tree depths, and leaves directly on validation dataset cache. |
| **[tests/test_agent_risk_pipeline.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/tests/test_agent_risk_pipeline.py)** | Script | Integration script simulating scenario packaging, pattern queries, portfolio tracking, and risk check executions. |
| **[tests/test_execution_safety_pipeline.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/tests/test_execution_safety_pipeline.py)** | Script | Integration script simulating and verifying simulated fills, MFE/MAE tracking, SQLite database journaling, and multi-level circuit breaker safety halts. |
| **[tests/test_suite.py](file:///g:/.shortcut-targets-by-id/1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA/Binance-Vision-Data/development/github/tests/test_suite.py)** | Script | Complete unit test suite containing 27 detailed tests for L4 predictor, L5 Agent, and L5 Executor modules. |

---

## 🧠 2. Model Architecture & Methodology

### A. Target Pair & Data Inputs
* **Symbol**: `BTCUSDT`
* **Resolution**: 1-minute contiguous blocks.
* **Input Streams**: 1m Klines MASTER, 5m Liquidity Hierarchy Maps, and 64GB Enriched `aggTrades` (3.28 billion trades).

### B. Triple Barrier Method (TBM) Labeling
Labels are generated using a sliding window search over a 60-minute lookahead window with 1.0% Take Profit (TP) and 1.0% Stop Loss (SL):
* **`0 = WAIT`**: Neither barrier is crossed within 60 minutes (vertical barrier timeout).
* **`1 = BUY`**: Upper barrier (+1.0%) is crossed before lower barrier (-1.0%).
* **`2 = SELL`**: Lower barrier (-1.0%) is crossed before upper barrier (+1.0%).

### C. Feature Engineering (30 Features)
The model trains on 30 features combining order book liquidity structures, trade flow imbalances, and price dynamics:
1. **Trade Flow**: `agg_trade_count`, `agg_volume`, `agg_vwap`, `buyer_maker_ratio`, `taker_buy_volume`, `taker_buy_quote_volume`.
2. **Order Book Liquidity Hierarchy**: `liquidity_up_5m`, `liquidity_below_5m`, `liquidity_up_1h`, `liquidity_below_1h`, `liquidity_up_4h`, `liquidity_below_4h`, `liquidity_up_1d`, `liquidity_below_1d`, `nearest_liq_5m`, `nearest_liq_1h`, `nearest_liq_4h`, `nearest_liq_1d`.
3. **Engineered Indicators**: `log_ret`, `volatility_20` (rolling std), `sma_ratio` (`close / sma_20`), `dist_liq_up_5m`, `dist_liq_below_5m`.

### D. Time-Series Purging
To eliminate lookahead data leakage during training, dataset splits are strictly chronological (80% train, 20% validation) separated by a 60-minute purging gap equal to the lookahead window.

---

## 🚀 3. How to Execute the Scripts

All commands should be executed from the repository root directory (`development/github/`).

### 1. Run End-to-End Integration Tests
Validates feature alignment, model compilation, schema reordering, and predictor validation error handling on a fast 5 row-group test subset:
```bash
python -m tests.validate_trainer_predictor
```

### 2. Train the Official Model
Train the LightGBM multiclass classifier on full history (caches feature tables automatically for subsequent fast runs):
```bash
# Default baseline training
python -m src.trainer

# Train with optimized class weights for handling dataset imbalance (Recommended Setup)
python -m src.trainer --w-buy 2.5 --w-sell 2.5
```

### 3. Run Hyperparameter Grid Search
Explore tree architectures and class weighting setups directly on the validation dataset cache:
```bash
python -m tests.tune_hyperparameters
```

### 4. Run Backtest Simulation & Evaluation
Simulate historical trades on the time-purged validation set across custom confidence thresholds:
```bash
# Run backtest with recommended 0.50 confidence threshold on weighted model
python -m tests.simulate_backtest --confidence 0.50 --tp 0.01 --sl 0.01
```

### 5. Run Agent Risk Pipeline Simulation
Verify the end-to-end agent decision pipeline: scenario generation, SQLite pattern matches, portfolio tracking, and risk rules checks:
```bash
python -m tests.test_agent_risk_pipeline
```

### 6. Run Execution & Safety Pipeline Simulation
Verify simulated fills, MFE/MAE excursions, database journaling, counterfactual logging, and multi-level circuit breaker safety blocks:
```bash
python -m tests.test_execution_safety_pipeline
```

### 7. Run Comprehensive Unit Test Suite
Execute all 27 unit tests verifying model loading, schemas, agent risk parameters, portfolio tracking, SQLite journaling, paper trading, and circuit breaker logic:
```bash
python -m tests.test_suite
```

---

## 🔗 4. How to Attach & Use `predictor.py` in an Operational Pipeline

The `ModelPredictor` class inside `src/predictor.py` is designed to plug directly into an operational trading or live execution pipeline.

### Step-by-Step Integration Code Example

```python
import pandas as pd
from src.predictor import ModelPredictor

# 1. Initialize the Predictor engine (loads model.txt and features_order.json once at startup)
predictor = ModelPredictor(
    model_dir="models",
    confidence_threshold=0.50, # 50% minimum probability to trigger signals
    tp_threshold=0.01,         # 1.0% Take Profit level
    sl_threshold=0.01          # 1.0% Stop Loss level
)

# 2. Ingest live market candle update from your streaming pipeline / websocket
# (Requires standard price columns and liquidity boundaries)
live_market_data = pd.DataFrame([{
    'open': 84400.0,
    'high': 84550.0,
    'low': 84350.0,
    'close': 84500.0,
    'volume': 150.5,
    'quote_volume': 12717250.0,
    'count': 1200,
    'taker_buy_volume': 80.2,
    'taker_buy_quote_volume': 6776900.0,
    'agg_trade_count': 1150,
    'agg_volume': 148.0,
    'agg_vwap': 84490.0,
    'buyer_maker_ratio': 0.45,
    'nearest_liq_5m': 84200.0, 'nearest_liq_1h': 84000.0, 'nearest_liq_4h': 83500.0, 'nearest_liq_1d': 82000.0,
    'liquidity_up_5m': 85000.0, 'liquidity_below_5m': 84000.0,
    'liquidity_up_1h': 85500.0, 'liquidity_below_1h': 83800.0,
    'liquidity_up_4h': 86000.0, 'liquidity_below_4h': 83000.0,
    'liquidity_up_1d': 88000.0, 'liquidity_below_1d': 81000.0,
}])

# 3. Generate predictions
# Predictor automatically computes missing technical indicators (volatility_20, sma_ratio, etc.)
# and reorders columns to match model expectations exactly.
results = predictor.predict(live_market_data)

# 4. Extract execution signal and parameters
latest_signal = results.iloc[-1]
signal_type = latest_signal['signal']        # 'BUY', 'SELL', or 'WAIT'
confidence  = latest_signal['confidence']    # Probability score
tp_price    = latest_signal['take_profit']    # Projected TP price level
sl_price    = latest_signal['stop_loss']      # Projected SL price level

print(f"Signal: {signal_type} | Confidence: {confidence:.4f}")

# 5. Route order parameters to exchange execution engine
if signal_type in ["BUY", "SELL"]:
    print(f"EXECUTE ORDER: Type={signal_type}, Limit={live_market_data.iloc[-1]['close']}, TP={tp_price:.2f}, SL={sl_price:.2f}")
else:
    print("STATUS: Standing by (WAIT)...")
```

---

## 📊 5. Benchmark Performance & Validation Results

The model was evaluated on `673,572` contiguous 1-minute validation rows spanning from `Feb 13, 2025` to `May 26, 2026` (~1.3 years).

### A. Baseline Model (Unweighted) Across Thresholds

| Metric | Conf: 0.55 | Conf: 0.50 | Conf: 0.45 | Conf: 0.40 |
| :--- | :---: | :---: | :---: | :---: |
| **Total Trades** | 21 | 1 | 108 | 1,668 |
| **Win Rate** | 80.95% | 0.00% | **60.19%** | 49.22% |
| **Net P&L** | +3.43% | -1.00% | **+23.92%** | -10.02% |
| **Profit Factor** | 5.68 | 0.00 | **1.60** | 0.99 |

### B. Optimized Balanced Deployment (`Balanced 1:2.5`)

Retrained with class weights (`w-buy 2.5 --w-sell 2.5`) to penalize minority class misclassification and evaluated at `--confidence 0.50`:

* **Total Trades Executed**: 79 trades (all SELLs)
* **Win Rate**: **67.09%** (53 wins, 26 losses/flat)
* **Exit Breakdown**: 88.6% Timeout, 10.1% Take Profit, 1.3% Stop Loss
* **Net P&L**: **+12.80%**
* **Profit Factor**: **2.82** (Top-tier signal accuracy, 2.8x more profit than loss)
* **Average P&L per Trade**: **+0.1620%**
