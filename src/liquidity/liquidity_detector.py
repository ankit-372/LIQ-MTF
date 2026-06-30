"""
Liquidity Detector (L3)
Detects structural liquidity levels where stop-losses cluster.
"""
from typing import Dict, List, Any
from src.core import event_bus

class Level:
    def __init__(self, price: float, type: str, tags: set, created_at_bar: int):
        self.price = price
        self.type = type  # 'high' or 'low'
        self.tags = set(tags)  # e.g., {'5m', '1h'}
        self.created_at_bar = created_at_bar
        self.active = True

class PivotDetector:
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.window: List[Dict[str, Any]] = []
        self.bar_count = 0
        
    def add_candle(self, candle: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Adds a candle and returns any detected pivots.
        A pivot is a dict: {'price': float, 'type': 'high'|'low', 'bar_index': int}
        """
        self.window.append(candle)
        self.bar_count += 1
        
        if len(self.window) > 11:
            self.window.pop(0)
            
        pivots = []
        if len(self.window) == 11:
            pivot_candle = self.window[5]
            
            # Check Pivot High
            high = float(pivot_candle['h'])
            left_highs = [float(c['h']) for c in self.window[0:5]]
            right_highs = [float(c['h']) for c in self.window[6:11]]
            if high > max(left_highs) and high > max(right_highs):
                pivots.append({
                    'price': high,
                    'type': 'high',
                    'bar_index': self.bar_count - 6  # Index of the pivot candle itself
                })
                
            # Check Pivot Low
            low = float(pivot_candle['l'])
            left_lows = [float(c['l']) for c in self.window[0:5]]
            right_lows = [float(c['l']) for c in self.window[6:11]]
            if low < min(left_lows) and low < min(right_lows):
                pivots.append({
                    'price': low,
                    'type': 'low',
                    'bar_index': self.bar_count - 6
                })
                
        return pivots

class LevelManager:
    def __init__(self):
        self.active_levels: List[Level] = []
        self.last_level_bar: int = -18  # Initialize so first level can be created
        
    def process_pivot(self, pivot: Dict[str, Any], current_bar: int, current_atr: float, timeframe: str):
        """
        Processes a newly detected pivot and creates/merges/rejects levels.
        """
        price = pivot['price']
        
        # 1. Clustering check (0.08 * ATR)
        merge_threshold = 0.08 * current_atr
        for level in self.active_levels:
            if abs(level.price - price) <= merge_threshold:
                level.tags.add(timeframe)
                return  # Merged, we are done
                
        # 2. Quality check (0.70 * ATR)
        reject_threshold = 0.70 * current_atr
        for level in self.active_levels:
            if abs(level.price - price) <= reject_threshold:
                return  # Rejected due to proximity to another level
                
        # 3. Temporal separation (18 bars)
        if (current_bar - self.last_level_bar) < 18:
            return  # Rejected due to temporal proximity
            
        # Create new level
        new_level = Level(
            price=price,
            type=pivot['type'],
            tags={timeframe},
            created_at_bar=current_bar
        )
        self.active_levels.append(new_level)
        self.last_level_bar = current_bar
        event_bus.publish("LEVEL_CREATED", {"price": price, "type": pivot['type'], "timeframe": timeframe})

class SweepDetector:
    def __init__(self, level_manager: LevelManager):
        self.level_manager = level_manager
        self.recent_sweeps = []  # List of bar indices when sweeps occurred
        
    def check_sweeps(self, candle: Dict[str, Any], current_bar: int):
        """
        Checks if the current candle sweeps any active levels.
        """
        high = float(candle['h'])
        low = float(candle['l'])
        
        unswept_levels = []
        swept_count_this_bar = 0
        
        for level in self.level_manager.active_levels:
            swept = False
            if level.type == 'high' and high >= level.price:
                swept = True
            elif level.type == 'low' and low <= level.price:
                swept = True
                
            if swept:
                level.active = False
                swept_count_this_bar += 1
                event_bus.publish("LEVEL_SWEPT", {
                    "price": level.price,
                    "type": level.type,
                    "tags": list(level.tags),
                    "sweep_bar": current_bar
                })
            else:
                unswept_levels.append(level)
                
        self.level_manager.active_levels = unswept_levels
        
        if swept_count_this_bar > 0:
            for _ in range(swept_count_this_bar):
                self.recent_sweeps.append(current_bar)
                
        # Cleanup recent sweeps older than 12 bars
        self.recent_sweeps = [b for b in self.recent_sweeps if (current_bar - b) <= 12]

class LiquidityDetector:
    def __init__(self):
        self.pivot_detectors = {
            '5m': PivotDetector('5m'),
            '1h': PivotDetector('1h'),
            '4h': PivotDetector('4h')
        }
        self.level_manager = LevelManager()
        self.sweep_detector = SweepDetector(self.level_manager)
        
        # Subscribe to candle events
        event_bus.subscribe("CANDLE_CLOSED_5M", self._handle_5m_candle)
        event_bus.subscribe("CANDLE_CLOSED_1H", self._handle_1h_candle)
        event_bus.subscribe("CANDLE_CLOSED_4H", self._handle_4h_candle)
        
        # State
        self.current_bar_5m = 0
        self.current_atr = 100.0  # Should be updated dynamically in a real system
        
    def _process_candle(self, candle: Dict[str, Any], timeframe: str):
        if timeframe == '5m':
            self.current_bar_5m += 1
            
        detector = self.pivot_detectors[timeframe]
        pivots = detector.add_candle(candle)
        
        # Process new pivots (using 5m bar count as the global time reference)
        for pivot in pivots:
            self.level_manager.process_pivot(
                pivot=pivot,
                current_bar=self.current_bar_5m,
                current_atr=self.current_atr,
                timeframe=timeframe
            )
            
        # Check for sweeps on every 5m candle
        if timeframe == '5m':
            self.sweep_detector.check_sweeps(candle, self.current_bar_5m)
            
    def _handle_5m_candle(self, candle: Dict[str, Any]):
        self._process_candle(candle, '5m')
        
    def _handle_1h_candle(self, candle: Dict[str, Any]):
        self._process_candle(candle, '1h')
        
    def _handle_4h_candle(self, candle: Dict[str, Any]):
        self._process_candle(candle, '4h')
        
    def set_atr(self, atr: float):
        """Method to update ATR from Indicator Engine."""
        self.current_atr = atr
        
    def get_features(self, current_price: float) -> Dict[str, float]:
        """
        Returns the liquidity features for the Feature Engine.
        """
        features = {
            'lvl_dist_above_5m': 999.0, 'lvl_dist_below_5m': 999.0,
            'lvl_dist_above_1h': 999.0, 'lvl_dist_below_1h': 999.0,
            'lvl_dist_above_4h': 999.0, 'lvl_dist_below_4h': 999.0,
            'lvl_active_count': float(len(self.level_manager.active_levels)),
            'lvl_recent_sweep_count': float(len(self.sweep_detector.recent_sweeps)),
            'lvl_nearest_confluence': 0.0  # 1.0 if nearest is confluence, else 0.0
        }
        
        # Calculate distances per timeframe
        for tf in ['5m', '1h', '4h']:
            above = [lvl for lvl in self.level_manager.active_levels if tf in lvl.tags and lvl.price > current_price]
            below = [lvl for lvl in self.level_manager.active_levels if tf in lvl.tags and lvl.price < current_price]
            
            if above:
                min_above = min(above, key=lambda x: x.price)
                features[f'lvl_dist_above_{tf}'] = ((min_above.price - current_price) / current_price) * 100
                
            if below:
                max_below = max(below, key=lambda x: x.price)
                features[f'lvl_dist_below_{tf}'] = ((current_price - max_below.price) / current_price) * 100
                
        # Nearest confluence check
        all_levels = self.level_manager.active_levels
        if all_levels:
            nearest = min(all_levels, key=lambda x: abs(x.price - current_price))
            if len(nearest.tags) > 1:
                features['lvl_nearest_confluence'] = 1.0
                
        return features
