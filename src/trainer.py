import os
import json
import time
import datetime
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix

class ModelTrainer:
    def __init__(
        self,
        klines_path: str,
        aggtrades_path: str,
        liquidity_path: str,
        cache_dir: str = "cache",
        model_dir: str = "models",
        tp_threshold: float = 0.01,
        sl_threshold: float = 0.01,
        lookahead_window: int = 60,
        train_val_split_ratio: float = 0.80,
        max_row_groups: int = None,
        # Hyperparameters
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        class_weight_wait: float = 1.0,
        class_weight_buy: float = 1.0,
        class_weight_sell: float = 1.0
    ):
        self.klines_path = klines_path
        self.aggtrades_path = aggtrades_path
        self.liquidity_path = liquidity_path
        self.cache_dir = cache_dir
        self.model_dir = model_dir
        self.tp_threshold = tp_threshold
        self.sl_threshold = sl_threshold
        self.lookahead_window = lookahead_window
        self.train_val_split_ratio = train_val_split_ratio
        self.max_row_groups = max_row_groups
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.class_weight_wait = class_weight_wait
        self.class_weight_buy = class_weight_buy
        self.class_weight_sell = class_weight_sell

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def aggregate_and_align_features(self) -> pd.DataFrame:
        """
        Loads the three datasets, aggregates aggTrades to 1-minute blocks in a memory-safe
        chunked way, and joins them on timestamp. Saves and loads from cache if available.
        """
        cache_file = os.path.join(self.cache_dir, "BTCUSDT_1m_aggregated_features.parquet")
        if os.path.exists(cache_file):
            print(f"Loading pre-aggregated features from cache: {cache_file}")
            df_aligned = pd.read_parquet(cache_file)
            # Ensure time columns are datetime
            df_aligned['datetime'] = pd.to_datetime(df_aligned['datetime'])
            df_aligned = df_aligned.set_index('datetime').sort_index()
            return df_aligned

        print("No cache found. Starting chunked aggregation of aggTrades (may take a few minutes)...")
        
        # 1. Read Klines (which is 100% contiguous 1-minute data)
        print(f"Reading klines from: {self.klines_path}")
        df_klines = pd.read_parquet(self.klines_path)
        df_klines['datetime'] = pd.to_datetime(df_klines['datetime'], unit='ms')
        df_klines = df_klines.set_index('datetime').sort_index()
        
        # 2. Read Liquidity Hierarchy Map (5-minute resolution)
        print(f"Reading liquidity hierarchy map from: {self.liquidity_path}")
        df_liq = pd.read_parquet(self.liquidity_path)
        # Normalize timezone to naive UTC (or timezone-aware, keeping consistent)
        df_liq['transact_time'] = pd.to_datetime(df_liq['transact_time']).dt.tz_localize(None)
        df_liq = df_liq.set_index('transact_time').sort_index()
        
        # Resample liquidity map to 1-minute frequency and forward-fill
        df_liq_1m = df_liq.resample('1min').ffill()
        
        # 3. Read and Aggregate aggTrades chunk-by-chunk using PyArrow
        print(f"Reading and aggregating aggTrades from: {self.aggtrades_path}")
        pf = pq.ParquetFile(self.aggtrades_path)
        num_row_groups = pf.num_row_groups
        if self.max_row_groups is not None:
            num_row_groups = min(num_row_groups, self.max_row_groups)
        
        agg_chunks = []
        start_time = time.time()
        
        # We read row groups of size 1 million.
        for rg_idx in range(num_row_groups):
            if rg_idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Processing row group {rg_idx}/{num_row_groups} ({rg_idx / num_row_groups * 100:.1f}% done, elapsed: {elapsed:.1f}s)...")
                
            # Read single row group
            rg_table = pf.read_row_group(
                rg_idx, 
                columns=['price', 'quantity', 'is_buyer_maker', 'transact_time', 'nearest_liq_5m', 'nearest_liq_1h', 'nearest_liq_4h', 'nearest_liq_1d']
            )
            df_rg = rg_table.to_pandas()
            
            # Convert transact_time to naive datetime
            df_rg['transact_time'] = pd.to_datetime(df_rg['transact_time']).dt.tz_localize(None)
            
            # Floor time to minute
            df_rg['datetime_1m'] = df_rg['transact_time'].dt.floor('1min')
            
            # Group by 1-minute intervals and aggregate
            # Note: is_buyer_maker=True represents taker-sell (maker-buy), is_buyer_maker=False represents taker-buy
            df_rg['weighted_price'] = df_rg['price'] * df_rg['quantity']
            df_rg['buy_qty'] = np.where(df_rg['is_buyer_maker'] == False, df_rg['quantity'], 0.0)
            df_rg['sell_qty'] = np.where(df_rg['is_buyer_maker'] == True, df_rg['quantity'], 0.0)
            
            # Aggregate group
            agg_rg = df_rg.groupby('datetime_1m').agg(
                agg_trade_count=('price', 'count'),
                agg_volume=('quantity', 'sum'),
                weighted_price_sum=('weighted_price', 'sum'),
                agg_buy_volume=('buy_qty', 'sum'),
                agg_sell_volume=('sell_qty', 'sum'),
                nearest_liq_5m=('nearest_liq_5m', 'last'),
                nearest_liq_1h=('nearest_liq_1h', 'last'),
                nearest_liq_4h=('nearest_liq_4h', 'last'),
                nearest_liq_1d=('nearest_liq_1d', 'last')
            )
            agg_chunks.append(agg_rg)

        # Concatenate and combine all chunks
        print("Concatenating aggregated trade chunks...")
        df_agg = pd.concat(agg_chunks)
        
        # Since the same minute can span across row group boundaries, group again on the combined dataframe
        print("Finalizing 1-minute trade aggregation...")
        df_agg = df_agg.groupby('datetime_1m').agg(
            agg_trade_count=('agg_trade_count', 'sum'),
            agg_volume=('agg_volume', 'sum'),
            weighted_price_sum=('weighted_price_sum', 'sum'),
            agg_buy_volume=('agg_buy_volume', 'sum'),
            agg_sell_volume=('agg_sell_volume', 'sum'),
            nearest_liq_5m=('nearest_liq_5m', 'last'),
            nearest_liq_1h=('nearest_liq_1h', 'last'),
            nearest_liq_4h=('nearest_liq_4h', 'last'),
            nearest_liq_1d=('nearest_liq_1d', 'last')
        )
        
        # Calculate VWAP
        df_agg['agg_vwap'] = np.where(
            df_agg['agg_volume'] > 0,
            df_agg['weighted_price_sum'] / df_agg['agg_volume'],
            np.nan
        )
        
        # Calculate buyer maker (taker sell) ratio
        df_agg['buyer_maker_ratio'] = np.where(
            df_agg['agg_volume'] > 0,
            df_agg['agg_sell_volume'] / df_agg['agg_volume'],
            0.5
        )
        
        # Drop temporary columns
        df_agg = df_agg.drop(columns=['weighted_price_sum', 'agg_buy_volume', 'agg_sell_volume'])
        
        # 4. Join all datasets on the 1-minute index
        print("Joining klines, trade aggregates, and liquidity map...")
        # Reindex trade aggregates to match the klines time index exactly, and forward fill missing trades
        df_agg_reindexed = df_agg.reindex(df_klines.index)
        df_agg_reindexed['agg_trade_count'] = df_agg_reindexed['agg_trade_count'].fillna(0)
        df_agg_reindexed['agg_volume'] = df_agg_reindexed['agg_volume'].fillna(0)
        df_agg_reindexed['buyer_maker_ratio'] = df_agg_reindexed['buyer_maker_ratio'].fillna(0.5)
        # Forward fill other trade features (VWAP, liquidity levels)
        df_agg_reindexed['agg_vwap'] = df_agg_reindexed['agg_vwap'].ffill().bfill()
        df_agg_reindexed['nearest_liq_5m'] = df_agg_reindexed['nearest_liq_5m'].ffill().bfill()
        df_agg_reindexed['nearest_liq_1h'] = df_agg_reindexed['nearest_liq_1h'].ffill().bfill()
        df_agg_reindexed['nearest_liq_4h'] = df_agg_reindexed['nearest_liq_4h'].ffill().bfill()
        df_agg_reindexed['nearest_liq_1d'] = df_agg_reindexed['nearest_liq_1d'].ffill().bfill()

        # Join everything
        df_aligned = df_klines.join(df_agg_reindexed, rsuffix='_trade')
        df_aligned = df_aligned.join(df_liq_1m, rsuffix='_liq')
        
        # Clean any remaining NaNs (e.g. from the beginning of datasets)
        df_aligned = df_aligned.ffill().bfill()
        
        # Save to cache
        print(f"Saving aggregated features to cache: {cache_file}")
        df_aligned.reset_index().to_parquet(cache_file)
        
        return df_aligned

    def generate_triple_barrier_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes Triple Barrier Method labels for each timestamp:
        - 1: BUY (if Take Profit hit first)
        - 2: SELL (if Stop Loss hit first)
        - 0: WAIT (if Lookahead Window expires)
        """
        print("Generating Triple Barrier Labels...")
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        n = len(df)
        
        labels = np.zeros(n, dtype=int)
        
        # Parameters
        pt = self.tp_threshold
        sl = self.sl_threshold
        w = self.lookahead_window
        
        # Optimized loop to look forward
        for i in range(n - w):
            p = close[i]
            p_up = p * (1.0 + pt)
            p_down = p * (1.0 - sl)
            
            # Slices for the lookahead window
            h_slice = high[i+1 : i+1+w]
            l_slice = low[i+1 : i+1+w]
            
            # Check crossings
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
                
        # Fill the final window elements as WAIT (0) due to insufficient lookahead
        labels[n - w :] = 0
        return pd.Series(labels, index=df.index)

    def train(self):
        """
        Coordinates feature loading, label generation, data purging/splitting,
        LightGBM model fitting, and saving model results.
        """
        # 1. Load and align features
        df = self.aggregate_and_align_features()
        
        # 2. Generate labels
        df['label'] = self.generate_triple_barrier_labels(df)
        
        # 3. Create target and feature sets
        # Define features
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'count',
            'taker_buy_volume', 'taker_buy_quote_volume',
            'agg_trade_count', 'agg_volume', 'agg_vwap', 'buyer_maker_ratio',
            'nearest_liq_5m', 'nearest_liq_1h', 'nearest_liq_4h', 'nearest_liq_1d',
            'liquidity_up_5m', 'liquidity_below_5m', 'liquidity_up_1h', 'liquidity_below_1h',
            'liquidity_up_4h', 'liquidity_below_4h', 'liquidity_up_1d', 'liquidity_below_1d'
        ]
        
        # Add basic feature engineering: rolling returns, log volatility, SMA ratio
        print("Engineering technical indicators...")
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_20'] = df['log_ret'].rolling(20).std()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_ratio'] = df['close'] / df['sma_20']
        
        # Price relation to liquidity metrics
        df['dist_liq_up_5m'] = (df['liquidity_up_5m'] - df['close']) / df['close']
        df['dist_liq_below_5m'] = (df['close'] - df['liquidity_below_5m']) / df['close']
        
        df = df.ffill().bfill()
        
        engineered_cols = ['log_ret', 'volatility_20', 'sma_ratio', 'dist_liq_up_5m', 'dist_liq_below_5m']
        all_features = feature_cols + engineered_cols
        
        # Keep exact feature name ordering
        feature_ordering = sorted(all_features)
        
        # 4. Strictly Time-Based purged split to prevent lookahead leakage
        print("Performing time-based train/validation split with purging...")
        total_len = len(df)
        split_idx = int(total_len * self.train_val_split_ratio)
        
        train_end_time = df.index[split_idx]
        val_start_time = train_end_time + pd.Timedelta(minutes=self.lookahead_window)
        
        # Purged split
        train_df = df.loc[:train_end_time]
        val_df = df.loc[val_start_time:]
        
        print(f"Train set: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)")
        print(f"Val set: {val_df.index.min()} to {val_df.index.max()} ({len(val_df)} rows)")
        print(f"Purging gap: {self.lookahead_window} minutes ({train_end_time} to {val_start_time})")
        
        X_train = train_df[feature_ordering]
        y_train = train_df['label']
        
        X_val = val_df[feature_ordering]
        y_val = val_df['label']
        
        print("Label distributions:")
        print(f"  Train labels:\n{y_train.value_counts(normalize=True)}")
        print(f"  Val labels:\n{y_val.value_counts(normalize=True)}")
        
        # 5. Train LightGBM multiclass classifier
        print("Fitting LightGBM multiclass model...")
        
        # Apply class weights to training dataset
        class_weights = {0: self.class_weight_wait, 1: self.class_weight_buy, 2: self.class_weight_sell}
        sample_weights = y_train.map(class_weights).values
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_jobs': -1,
            'random_state': 42,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'verbose': -1
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        # 6. Evaluation metrics
        print("Running evaluations on validation set...")
        y_pred_probs = model.predict(X_val)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        report_str = classification_report(y_val, y_pred, digits=4)
        conf_mat = confusion_matrix(y_val, y_pred)
        
        print("Classification Report on Validation set:")
        print(report_str)
        print("Confusion Matrix:")
        print(conf_mat)
        
        # Feature importances
        importance = model.feature_importance(importance_type='gain')
        feat_imp_df = pd.DataFrame({
            'feature': feature_ordering,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 features by Gain:")
        print(feat_imp_df.head(10))
        
        # 7. Save trained artifacts
        model_file = os.path.join(self.model_dir, "model.txt")
        model.save_model(model_file)
        print(f"Saved model to: {model_file}")
        
        feat_order_file = os.path.join(self.model_dir, "features_order.json")
        with open(feat_order_file, 'w') as f:
            json.dump(feature_ordering, f, indent=4)
        print(f"Saved feature ordering to: {feat_order_file}")
        
        # Save metadata
        metadata = {
            "training_run_time": datetime.datetime.utcnow().isoformat() + "Z",
            "hyperparameters": params,
            "train_set_range": [str(train_df.index.min()), str(train_df.index.max())],
            "val_set_range": [str(val_df.index.min()), str(val_df.index.max())],
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "label_distribution_train": y_train.value_counts().to_dict(),
            "label_distribution_val": y_val.value_counts().to_dict(),
            "confusion_matrix": conf_mat.tolist(),
            "validation_report": classification_report(y_val, y_pred, digits=4, output_dict=True)
        }
        
        metadata_file = os.path.join(self.model_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved training metadata to: {metadata_file}")
        
        # Save evaluations text report
        report_file = os.path.join(self.model_dir, "evaluation_report.txt")
        with open(report_file, 'w') as f:
            f.write("=== CLASSIFICATION REPORT ===\n")
            f.write(report_str)
            f.write("\n=== CONFUSION MATRIX ===\n")
            f.write(str(conf_mat) + "\n")
            f.write("\n=== FEATURE IMPORTANCE (GAIN) ===\n")
            f.write(feat_imp_df.to_string(index=False) + "\n")
        print(f"Saved evaluation report text to: {report_file}")
        
        print("Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model Trainer for L4 Predictor")
    parser.add_argument("--klines", type=str, default=r"g:\.shortcut-targets-by-id\1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA\Binance-Vision-Data\inspect-data-iamrao\BTCUSDT_1m_klines_MASTER.parquet")
    parser.add_argument("--aggtrades", type=str, default=r"g:\.shortcut-targets-by-id\1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA\Binance-Vision-Data\ML-Ready_Data\aggTrades\BTCUSDT\BTCUSDT_aggTrades_Enriched_MASTER.parquet")
    parser.add_argument("--liquidity", type=str, default=r"g:\.shortcut-targets-by-id\1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA\Binance-Vision-Data\ML-Ready_Data\Liquidity_Engine\BTCUSDT\BTCUSDT_Liquidity_Hierarchy_Map.parquet")
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--tp", type=float, default=0.01)
    parser.add_argument("--sl", type=float, default=0.01)
    parser.add_argument("--lookahead", type=int, default=60)
    parser.add_argument("--max-row-groups", type=int, default=None)
    # Hyperparameter parser arguments
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=0.0)
    parser.add_argument("--w-wait", type=float, default=1.0)
    parser.add_argument("--w-buy", type=float, default=1.0)
    parser.add_argument("--w-sell", type=float, default=1.0)
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(
        klines_path=args.klines,
        aggtrades_path=args.aggtrades,
        liquidity_path=args.liquidity,
        cache_dir=args.cache_dir,
        model_dir=args.model_dir,
        tp_threshold=args.tp,
        sl_threshold=args.sl,
        lookahead_window=args.lookahead,
        max_row_groups=args.max_row_groups,
        learning_rate=args.lr,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        class_weight_wait=args.w_wait,
        class_weight_buy=args.w_buy,
        class_weight_sell=args.w_sell
    )
    trainer.train()
