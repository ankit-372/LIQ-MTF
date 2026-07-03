import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.shared.feature_registry import FEATURE_COLUMNS

class ModelPredictor:
    def __init__(
        self,
        model_dir: str = "models",
        confidence_threshold: float = 0.50,
        tp_threshold: float = None,
        sl_threshold: float = None
    ):
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold
        
        # Use feature ordering from registry
        self.feature_ordering = FEATURE_COLUMNS
            
        # Load model
        model_file = os.path.join(model_dir, "model.txt")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}. Run training first.")
        self.model = lgb.Booster(model_file=model_file)
        
        # Load thresholds from metadata if not explicitly provided
        metadata_file = os.path.join(model_dir, "metadata.json")
        self.tp_threshold = tp_threshold
        self.sl_threshold = sl_threshold
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                meta = json.load(f)
                train_meta = meta.get("hyperparameters", {})
                # Extract tp/sl from meta if they are none
                if self.tp_threshold is None:
                    # Default from trainer setup if stored, else fallback
                    self.tp_threshold = meta.get("tp_threshold", 0.01)
                if self.sl_threshold is None:
                    self.sl_threshold = meta.get("sl_threshold", 0.01)
        
        # Fallbacks
        if self.tp_threshold is None:
            self.tp_threshold = 0.01
        if self.sl_threshold is None:
            self.sl_threshold = 0.01

        print(f"Predictor initialized with model loaded from: {model_file}")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"TP Threshold: {self.tp_threshold}, SL Threshold: {self.sl_threshold}")

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to compute engineered indicators on-the-fly for live/future inference
        if the raw feature columns are provided.
        """
        df = df.copy()
        
        # Computes indicator variables required by the model
        if 'log_ret' not in df.columns:
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            
        if 'volatility_20' not in df.columns:
            df['volatility_20'] = df['log_ret'].rolling(20).std()
            
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(20).mean()
            
        if 'sma_ratio' not in df.columns:
            df['sma_ratio'] = df['close'] / df['sma_20']
            
        if 'dist_liq_up_5m' not in df.columns:
            df['dist_liq_up_5m'] = (df['liquidity_up_5m'] - df['close']) / df['close']
            
        if 'dist_liq_below_5m' not in df.columns:
            df['dist_liq_below_5m'] = (df['close'] - df['liquidity_below_5m']) / df['close']
            
        # Fill first few rows' NaNs if rolling windows didn't complete
        df = df.ffill().bfill()
        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs prediction on the incoming dataframe of features.
        Validates column names, types, and ordering.
        Computes probabilities, thresholds signals, and determines TP/SL values.
        """
        # Validate that it is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input features must be a pandas DataFrame.")
            
        # Compute features if some are missing but base columns are available
        missing_but_computable = [col for col in self.feature_ordering if col not in df.columns]
        computables = ['log_ret', 'volatility_20', 'sma_ratio', 'dist_liq_up_5m', 'dist_liq_below_5m']
        
        needed_computables = [col for col in computables if col in missing_but_computable]
        if needed_computables:
            # Check if required base columns are present
            base_required = []
            if any(x in needed_computables for x in ['log_ret', 'volatility_20', 'sma_ratio']):
                base_required.append('close')
            if 'dist_liq_up_5m' in needed_computables:
                base_required.append('liquidity_up_5m')
                if 'close' not in base_required:
                    base_required.append('close')
            if 'dist_liq_below_5m' in needed_computables:
                base_required.append('liquidity_below_5m')
                if 'close' not in base_required:
                    base_required.append('close')
                    
            missing_base = [col for col in base_required if col not in df.columns]
            if missing_base:
                raise ValueError(
                    f"Validation failed: Cannot compute technical indicators because the following required base columns are missing: {missing_base}. "
                    f"Expected features: {self.feature_ordering}"
                )
                
            print("Missing technical indicators. Computing on-the-fly...")
            df = self.compute_technical_indicators(df)
            
        # Re-check for missing columns
        missing_cols = [col for col in self.feature_ordering if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Validation failed: The input DataFrame is missing the following required feature columns: {missing_cols}. "
                f"Expected features: {self.feature_ordering}"
            )
            

        # Reorder columns to match the exact training feature order
        df_model = df[self.feature_ordering]
        
        # Run predict_proba (predict output will be probabilities since it is multiclass)
        pred_probs = self.model.predict(df_model) # Output shape: (N, 3)
        
        # Determine signals based on confidence threshold
        # Class 0: WAIT, Class 1: BUY, Class 2: SELL
        signals = []
        confidences = []
        tp_prices = []
        sl_prices = []
        
        close_prices = df['close'].values
        
        for i in range(len(df_model)):
            probs = pred_probs[i]
            # Max index
            max_class = int(np.argmax(probs))
            max_prob = float(probs[max_class])
            
            close_p = float(close_prices[i])
            
            # Apply confidence thresholding
            # Only trigger BUY (1) or SELL (2) if the probability exceeds confidence_threshold
            if max_class in [1, 2] and max_prob >= self.confidence_threshold:
                if max_class == 1:
                    signal = "BUY"
                    tp = close_p * (1.0 + self.tp_threshold)
                    sl = close_p * (1.0 - self.sl_threshold)
                else:
                    signal = "SELL"
                    tp = close_p * (1.0 - self.tp_threshold)
                    sl = close_p * (1.0 + self.sl_threshold)
            else:
                signal = "WAIT"
                tp = np.nan
                sl = np.nan
                max_prob = float(probs[0]) # probability of WAIT
                
            signals.append(signal)
            confidences.append(max_prob)
            tp_prices.append(tp)
            sl_prices.append(sl)
            
        # Output DataFrame
        res_df = pd.DataFrame({
            "signal": signals,
            "confidence": confidences,
            "take_profit": tp_prices,
            "stop_loss": sl_prices
        }, index=df.index)
        
        return res_df

if __name__ == "__main__":
    # Test instantiation
    try:
        predictor = ModelPredictor()
    except Exception as e:
        print("Initialization check (expected to fail if model not trained yet):", e)
