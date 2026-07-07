import sys
import os
import numpy as np
import pandas as pd

# Add repository root to sys.path to make imports work from any path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import ModelPredictor

def run_backtest(confidence_threshold: float = 0.45, tp_threshold: float = 0.01, sl_threshold: float = 0.01):
    print("==================================================")
    print("RUNNING BACKTEST SIMULATION ON VALIDATION SET")
    print("==================================================")
    
    # 1. Load Predictor
    model_dir = "models"
    predictor = ModelPredictor(
        model_dir=model_dir,
        confidence_threshold=confidence_threshold,
        tp_threshold=tp_threshold,
        sl_threshold=sl_threshold
    )
    
    # 2. Load aligned dataset from cache
    cache_file = "cache/BTCUSDT_1m_aggregated_features.parquet"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}. Run training first.")
        
    print(f"Loading aligned features from cache: {cache_file}")
    df = pd.read_parquet(cache_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    
    # 3. Split validation set
    # Using 80% train, 20% validation split index
    total_len = len(df)
    train_val_split_ratio = 0.80
    split_idx = int(total_len * train_val_split_ratio)
    
    train_end_time = df.index[split_idx]
    # Purging gap: 60 minutes
    val_start_time = train_end_time + pd.Timedelta(minutes=60)
    
    df_val = df.loc[val_start_time:].copy()
    print(f"Validation set size: {len(df_val)} rows (from {df_val.index.min()} to {df_val.index.max()})")
    
    # 4. Predict signals
    print("Generating predictions on validation set...")
    pred_res = predictor.predict(df_val)
    
    # Merge predictions with close/high/low price data for backtest
    backtest_df = df_val[['open', 'high', 'low', 'close']].join(pred_res)
    
    # 5. Simulate trades
    print("Simulating trades...")
    trades = []
    
    close_prices = backtest_df['close'].values
    high_prices = backtest_df['high'].values
    low_prices = backtest_df['low'].values
    signals = backtest_df['signal'].values
    confidences = backtest_df['confidence'].values
    tp_prices = backtest_df['take_profit'].values
    sl_prices = backtest_df['stop_loss'].values
    timestamps = backtest_df.index
    
    n = len(backtest_df)
    lookahead = 60 # 60 minute maximum trade duration (vertical barrier)
    
    for i in range(n - lookahead):
        sig = signals[i]
        if sig == "WAIT":
            continue
            
        entry_price = close_prices[i]
        entry_time = timestamps[i]
        tp_price = tp_prices[i]
        sl_price = sl_prices[i]
        
        # Look forward to find exit
        exit_p = None
        exit_time = None
        exit_reason = None
        trade_pnl = 0.0
        
        for j in range(i + 1, i + 1 + lookahead):
            high_j = high_prices[j]
            low_j = low_prices[j]
            time_j = timestamps[j]
            
            if sig == "BUY":
                # Check Stop Loss first (conservative check)
                if low_j <= sl_price:
                    exit_p = sl_price
                    exit_time = time_j
                    exit_reason = "STOP_LOSS"
                    trade_pnl = -predictor.sl_threshold
                    break
                # Check Take Profit
                elif high_j >= tp_price:
                    exit_p = tp_price
                    exit_time = time_j
                    exit_reason = "TAKE_PROFIT"
                    trade_pnl = predictor.tp_threshold
                    break
            elif sig == "SELL":
                # Check Stop Loss first (conservative check)
                if high_j >= sl_price:
                    exit_p = sl_price
                    exit_time = time_j
                    exit_reason = "STOP_LOSS"
                    trade_pnl = -predictor.sl_threshold
                    break
                # Check Take Profit
                elif low_j <= tp_price:
                    exit_p = tp_price
                    exit_time = time_j
                    exit_reason = "TAKE_PROFIT"
                    trade_pnl = predictor.tp_threshold
                    break
                    
        # If no horizontal barrier hit, exit at the close of the lookahead window (vertical barrier)
        if exit_reason is None:
            exit_idx = i + lookahead
            exit_p = close_prices[exit_idx]
            exit_time = timestamps[exit_idx]
            exit_reason = "TIMEOUT"
            if sig == "BUY":
                trade_pnl = (exit_p - entry_price) / entry_price
            else:
                trade_pnl = (entry_price - exit_p) / entry_price
                
        trades.append({
            "type": sig,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_p,
            "reason": exit_reason,
            "pnl": trade_pnl,
            "confidence": confidences[i]
        })
        
    # Convert to DataFrame
    df_trades = pd.DataFrame(trades)
    
    # 6. Evaluation metrics
    num_trades = len(df_trades)
    print(f"\nTotal trades executed: {num_trades}")
    if num_trades == 0:
        print("No trades triggered under the given confidence threshold.")
        return
        
    buy_trades = df_trades[df_trades['type'] == 'BUY']
    sell_trades = df_trades[df_trades['type'] == 'SELL']
    print(f"  BUY trades: {len(buy_trades)}")
    print(f"  SELL trades: {len(sell_trades)}")
    
    # Reasons for exit
    reasons = df_trades['reason'].value_counts()
    print("\nExit Reasons:")
    for reason, count in reasons.items():
        print(f"  {reason}: {count} ({count/num_trades*100:.1f}%)")
        
    # Win rate
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] < 0]
    win_rate = len(wins) / num_trades * 100
    print(f"\nWin Rate: {win_rate:.2f}% ({len(wins)} wins, {num_trades - len(wins)} losses/flat)")
    
    # Profits and Losses
    total_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    total_loss = df_trades[df_trades['pnl'] < 0]['pnl'].sum()
    net_pnl = df_trades['pnl'].sum()
    
    print(f"\nTotal Profit: {total_profit*100:.2f}%")
    print(f"Total Loss: {total_loss*100:.2f}%")
    print(f"Net P&L: {net_pnl*100:.2f}%")
    
    profit_factor = total_profit / abs(total_loss) if total_loss != 0 else np.inf
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average P&L per trade: {df_trades['pnl'].mean()*100:.4f}%")
    
    print("\nTop 5 most profitable trades:")
    print(df_trades.sort_values('pnl', ascending=False).head(5))
    
    print("\nTop 5 least profitable trades:")
    print(df_trades.sort_values('pnl').head(5))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate backtest on validation data.")
    parser.add_argument("--confidence", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--tp", type=float, default=0.01, help="Take profit threshold (default: 0.01)")
    parser.add_argument("--sl", type=float, default=0.01, help="Stop loss threshold (default: 0.01)")
    args = parser.parse_args()
    
    run_backtest(
        confidence_threshold=args.confidence,
        tp_threshold=args.tp,
        sl_threshold=args.sl
    )
