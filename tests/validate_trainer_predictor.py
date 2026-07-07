import sys
import os
import shutil
import numpy as np
import pandas as pd

# Add repository root to sys.path to make imports work from any path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer import ModelTrainer
from src.predictor import ModelPredictor

def run_integration_test():
    print("==================================================")
    print("STARTING END-TO-END VALIDATION TEST")
    print("==================================================")
    
    # 1. Paths
    klines_path = r"g:\.shortcut-targets-by-id\1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA\Binance-Vision-Data\inspect-data-iamrao\BTCUSDT_1m_klines_MASTER.parquet"
    aggtrades_path = r"g:\.shortcut-targets-by-id\1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA\Binance-Vision-Data\ML-Ready_Data\aggTrades\BTCUSDT\BTCUSDT_aggTrades_Enriched_MASTER.parquet"
    liquidity_path = r"g:\.shortcut-targets-by-id\1X1yx5zBlLBvjGIqomOuobRScrBJo9jEA\Binance-Vision-Data\ML-Ready_Data\Liquidity_Engine\BTCUSDT\BTCUSDT_Liquidity_Hierarchy_Map.parquet"
    
    test_cache_dir = "cache_test"
    test_model_dir = "models_test"
    
    # Clean previous test directories if they exist
    if os.path.exists(test_cache_dir):
        shutil.rmtree(test_cache_dir)
    if os.path.exists(test_model_dir):
        shutil.rmtree(test_model_dir)
        
    print("1. Running ModelTrainer on a subset (max_row_groups=5) to verify training pipeline...")
    # Instantiate trainer on 5 row groups
    trainer = ModelTrainer(
        klines_path=klines_path,
        aggtrades_path=aggtrades_path,
        liquidity_path=liquidity_path,
        cache_dir=test_cache_dir,
        model_dir=test_model_dir,
        tp_threshold=0.005, # 0.5% for test
        sl_threshold=0.005,
        lookahead_window=30, # 30 mins
        train_val_split_ratio=0.80,
        max_row_groups=5
    )
    
    # Train
    trainer.train()
    print("Trainer finished successfully! Check model output files.")
    
    # Verify saved files
    expected_files = ["model.txt", "features_order.json", "metadata.json", "evaluation_report.txt"]
    for f in expected_files:
        path = os.path.join(test_model_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing expected output model file: {path}")
        print(f"  Verified model output file exists: {path}")
        
    print("\n2. Initializing ModelPredictor and running inference...")
    # Load predictor using the test model directory
    predictor = ModelPredictor(
        model_dir=test_model_dir,
        confidence_threshold=0.45,
        tp_threshold=0.005,
        sl_threshold=0.005
    )
    
    # Load the test feature matrix from cache to get sample rows for inference
    cache_file = os.path.join(test_cache_dir, "BTCUSDT_1m_aggregated_features.parquet")
    df_features = pd.read_parquet(cache_file)
    
    # We select some rows for validation (take first 50 rows)
    df_sample = df_features.head(50).copy()
    
    # Run prediction
    res_df = predictor.predict(df_sample)
    
    print("\nSample predictions:")
    print(res_df.head(10))
    print("\nPrediction value counts:")
    print(res_df['signal'].value_counts())
    
    # Verify columns in predictions
    expected_pred_cols = ["signal", "confidence", "take_profit", "stop_loss"]
    for col in expected_pred_cols:
        if col not in res_df.columns:
            raise KeyError(f"Expected prediction column '{col}' missing in predictor output.")
            
    print("\n3. Verifying predictor validation strictness...")
    # Test 3a: missing column rejection
    df_missing = df_sample.drop(columns=["close"])
    try:
        predictor.predict(df_missing)
        raise RuntimeError("Predictor failed to reject input DataFrame with missing 'close' column!")
    except ValueError as e:
        print("  Success: Predictor correctly rejected missing column with error:", e)
        
    # Test 3b: wrong column ordering (should succeed because predictor automatically reorders columns)
    df_reordered = df_sample[sorted(df_sample.columns)]
    try:
        res_reordered = predictor.predict(df_reordered)
        print("  Success: Predictor correctly handled reordered columns and generated predictions.")
        # Check that outputs are identical (deterministic check)
        if not res_df.equals(res_reordered):
            raise ValueError("Predictor outputs are not identical for reordered columns!")
        print("  Success: Predictor predictions are identical (deterministic inference verified).")
    except Exception as e:
        raise RuntimeError(f"Predictor failed on reordered columns: {e}")

    # Clean up test directories after test passes
    print("\nCleaning up test artifacts...")
    shutil.rmtree(test_cache_dir)
    shutil.rmtree(test_model_dir)
    
    print("==================================================")
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_integration_test()
