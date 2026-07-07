import sys
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report

# Add repository root to sys.path to make imports work from any path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("==================================================")
    print("HYPERPARAMETER GRID SEARCH & TUNING TOOL")
    print("==================================================")
    
    cache_file = "cache/BTCUSDT_1m_aggregated_features.parquet"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}. Run trainer.py first to build features.")
        
    print(f"Loading aligned features from cache: {cache_file}...")
    df = pd.read_parquet(cache_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    
    # 1. Feature Engineering
    print("Computing technical features...")
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['log_ret'].rolling(20).std()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_ratio'] = df['close'] / df['sma_20']
    df['dist_liq_up_5m'] = (df['liquidity_up_5m'] - df['close']) / df['close']
    df['dist_liq_below_5m'] = (df['close'] - df['liquidity_below_5m']) / df['close']
    df = df.ffill().bfill()
    
    # 2. Label Generation (TP = 1.0%, SL = 1.0%, lookahead = 60 mins)
    print("Generating Triple Barrier labels...")
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(df)
    
    labels = np.zeros(n, dtype=int)
    pt = 0.01
    sl = 0.01
    w = 60
    
    for i in range(n - w):
        p = close[i]
        p_up = p * (1.0 + pt)
        p_down = p * (1.0 - sl)
        h_slice = high[i+1 : i+1+w]
        l_slice = low[i+1 : i+1+w]
        
        up_crossings = np.where(h_slice >= p_up)[0]
        down_crossings = np.where(l_slice <= p_down)[0]
        
        first_up = up_crossings[0] if len(up_crossings) > 0 else w
        first_down = down_crossings[0] if len(down_crossings) > 0 else w
        
        if first_up < first_down:
            labels[i] = 1 # BUY
        elif first_down < first_up:
            labels[i] = 2 # SELL
        else:
            labels[i] = 0 # WAIT
    labels[n - w :] = 0
    df['label'] = labels
    
    # 3. Purged Split
    total_len = len(df)
    train_val_split_ratio = 0.80
    split_idx = int(total_len * train_val_split_ratio)
    
    train_end_time = df.index[split_idx]
    val_start_time = train_end_time + pd.Timedelta(minutes=w)
    
    train_df = df.loc[:train_end_time]
    val_df = df.loc[val_start_time:].copy()
    
    # Features lists
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'count',
        'taker_buy_volume', 'taker_buy_quote_volume',
        'agg_trade_count', 'agg_volume', 'agg_vwap', 'buyer_maker_ratio',
        'nearest_liq_5m', 'nearest_liq_1h', 'nearest_liq_4h', 'nearest_liq_1d',
        'liquidity_up_5m', 'liquidity_below_5m', 'liquidity_up_1h', 'liquidity_below_1h',
        'liquidity_up_4h', 'liquidity_below_4h', 'liquidity_up_1d', 'liquidity_below_1d'
    ]
    engineered_cols = ['log_ret', 'volatility_20', 'sma_ratio', 'dist_liq_up_5m', 'dist_liq_below_5m']
    feature_ordering = sorted(feature_cols + engineered_cols)
    
    X_train = train_df[feature_ordering]
    y_train = train_df['label']
    X_val = val_df[feature_ordering]
    y_val = val_df['label']
    
    # Prepare val dataset for backtesting simulation
    close_prices = val_df['close'].values
    high_prices = val_df['high'].values
    low_prices = val_df['low'].values
    timestamps = val_df.index
    
    # Hyperparameter search space
    grid = [
        # (w_wait, w_buy, w_sell, max_depth, num_leaves, lr)
        {"w_wait": 1.0, "w_buy": 1.0, "w_sell": 1.0, "max_depth": 6, "num_leaves": 31, "lr": 0.05, "desc": "Baseline Unweighted"},
        {"w_wait": 1.0, "w_buy": 2.5, "w_sell": 2.5, "max_depth": 6, "num_leaves": 31, "lr": 0.05, "desc": "Balanced 1:2.5"},
        {"w_wait": 1.0, "w_buy": 5.0, "w_sell": 5.0, "max_depth": 6, "num_leaves": 31, "lr": 0.05, "desc": "Balanced 1:5"},
        {"w_wait": 1.0, "w_buy": 8.0, "w_sell": 8.0, "max_depth": 6, "num_leaves": 31, "lr": 0.05, "desc": "Balanced 1:8"},
        {"w_wait": 1.0, "w_buy": 5.0, "w_sell": 5.0, "max_depth": 4, "num_leaves": 15, "lr": 0.05, "desc": "Balanced 1:5, Shallow tree"},
        {"w_wait": 1.0, "w_buy": 5.0, "w_sell": 5.0, "max_depth": 8, "num_leaves": 63, "lr": 0.03, "desc": "Balanced 1:5, Deep tree"},
    ]
    
    results = []
    
    print("\nStarting Grid Search...")
    for idx, params in enumerate(grid):
        print(f"\n--- Running Config {idx + 1}/{len(grid)}: {params['desc']} ---")
        
        # Prepare datasets with class weights
        class_weights = {0: params["w_wait"], 1: params["w_buy"], 2: params["w_sell"]}
        sample_weights = y_train.map(class_weights).values
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_jobs': -1,
            'random_state': 42,
            'learning_rate': params["lr"],
            'max_depth': params["max_depth"],
            'num_leaves': params["num_leaves"],
            'verbose': -1
        }
        
        # Train model
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        
        # Validation predictions
        pred_probs = model.predict(X_val)
        
        # Run backtest simulation on various confidence thresholds
        # Let's test 0.40, 0.45, 0.50 confidence thresholds for each model!
        for conf_threshold in [0.40, 0.45, 0.50]:
            trades = []
            
            # Backtest loop
            for i in range(len(val_df) - w):
                probs = pred_probs[i]
                max_class = int(np.argmax(probs))
                max_prob = float(probs[max_class])
                
                if max_class in [1, 2] and max_prob >= conf_threshold:
                    sig = "BUY" if max_class == 1 else "SELL"
                    entry_p = close_prices[i]
                    entry_time = timestamps[i]
                    tp_p = entry_p * (1.0 + pt) if sig == "BUY" else entry_p * (1.0 - pt)
                    sl_p = entry_p * (1.0 - sl) if sig == "BUY" else entry_p * (1.0 + sl)
                    
                    # Exit scan
                    exit_p = None
                    exit_reason = None
                    trade_pnl = 0.0
                    
                    for j in range(i + 1, i + 1 + w):
                        high_j = high_prices[j]
                        low_j = low_prices[j]
                        
                        if sig == "BUY":
                            if low_j <= sl_p:
                                exit_p = sl_p
                                exit_reason = "STOP_LOSS"
                                trade_pnl = -sl
                                break
                            elif high_j >= tp_p:
                                exit_p = tp_p
                                exit_reason = "TAKE_PROFIT"
                                trade_pnl = pt
                                break
                        elif sig == "SELL":
                            if high_j >= sl_p:
                                exit_p = sl_p
                                exit_reason = "STOP_LOSS"
                                trade_pnl = -sl
                                break
                            elif low_j <= tp_p:
                                exit_p = tp_p
                                exit_reason = "TAKE_PROFIT"
                                trade_pnl = pt
                                break
                                
                    if exit_reason is None:
                        exit_idx = i + w
                        exit_p = close_prices[exit_idx]
                        exit_reason = "TIMEOUT"
                        if sig == "BUY":
                            trade_pnl = (exit_p - entry_p) / entry_p
                        else:
                            trade_pnl = (entry_p - exit_p) / entry_p
                            
                    trades.append(trade_pnl)
            
            # Compute trade metrics
            num_trades = len(trades)
            if num_trades > 0:
                wins = sum(1 for p in trades if p > 0)
                losses = sum(1 for p in trades if p < 0)
                win_rate = wins / num_trades * 100
                total_profit = sum(p for p in trades if p > 0)
                total_loss = sum(p for p in trades if p < 0)
                net_pnl = sum(trades) * 100
                profit_factor = total_profit / abs(total_loss) if total_loss != 0 else np.inf
            else:
                win_rate = 0.0
                net_pnl = 0.0
                profit_factor = 0.0
                
            print(f"  Conf: {conf_threshold:.2f} -> Trades: {num_trades}, Win Rate: {win_rate:.2f}%, Net P&L: {net_pnl:.2f}%, PF: {profit_factor:.2f}")
            
            results.append({
                "description": params["desc"],
                "w_buy": params["w_buy"],
                "max_depth": params["max_depth"],
                "num_leaves": params["num_leaves"],
                "learning_rate": params["lr"],
                "confidence_threshold": conf_threshold,
                "trades": num_trades,
                "win_rate": win_rate,
                "net_pnl": net_pnl,
                "profit_factor": profit_factor
            })
            
    # Output report
    print("\n==================================================")
    print("GRID SEARCH COMPLETED - RESULTS SUMMARY")
    print("==================================================")
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="net_pnl", ascending=False)
    print(df_res.to_string(index=False))
    
    best_config = df_res.iloc[0]
    print("\n==================================================")
    print(f"BEST CONFIGURATION FOUND:")
    print(f"Description: {best_config['description']}")
    print(f"Class Weights: Wait=1.0, BUY={best_config['w_buy']:.1f}, SELL={best_config['w_buy']:.1f}")
    print(f"Structure: max_depth={best_config['max_depth']}, num_leaves={best_config['num_leaves']}, lr={best_config['learning_rate']}")
    print(f"Best Confidence Threshold: {best_config['confidence_threshold']:.2f}")
    print(f"Trades Executed: {best_config['trades']}")
    print(f"Validation Win Rate: {best_config['win_rate']:.2f}%")
    print(f"Validation Net P&L: {best_config['net_pnl']:.2f}%")
    print(f"Profit Factor: {best_config['profit_factor']:.2f}")
    print("==================================================")

if __name__ == "__main__":
    main()
