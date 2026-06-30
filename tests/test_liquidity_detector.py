import pytest
from src.liquidity.liquidity_detector import PivotDetector, LevelManager, SweepDetector, LiquidityDetector
from src.core import event_bus

@pytest.fixture(autouse=True)
def setup_teardown():
    event_bus.clear_subscribers()
    yield
    event_bus.clear_subscribers()

def test_pivot_high_detection():
    detector = PivotDetector('5m')
    
    # 5 bars before pivot (lower highs)
    for i in range(5):
        detector.add_candle({'h': 100 + i, 'l': 90})
        
    # Pivot bar (highest high)
    detector.add_candle({'h': 110, 'l': 90})
    
    # 4 bars after pivot (lower highs)
    for i in range(4):
        pivots = detector.add_candle({'h': 105 - i, 'l': 90})
        assert len(pivots) == 0  # Pivot not confirmed yet
        
    # 5th bar after pivot (confirms it)
    pivots = detector.add_candle({'h': 100, 'l': 90})
    
    assert len(pivots) == 1
    assert pivots[0]['type'] == 'high'
    assert pivots[0]['price'] == 110
    
def test_pivot_low_detection():
    detector = PivotDetector('5m')
    
    # 5 bars before pivot (higher lows)
    for i in range(5):
        detector.add_candle({'h': 110, 'l': 100 - i})
        
    # Pivot bar (lowest low)
    detector.add_candle({'h': 110, 'l': 90})
    
    # 4 bars after pivot (higher lows)
    for i in range(4):
        pivots = detector.add_candle({'h': 110, 'l': 95 + i})
        assert len(pivots) == 0
        
    # 5th bar after pivot
    pivots = detector.add_candle({'h': 110, 'l': 100})
    
    assert len(pivots) == 1
    assert pivots[0]['type'] == 'low'
    assert pivots[0]['price'] == 90

def test_level_manager_clustering():
    manager = LevelManager()
    atr = 10.0
    
    # First pivot creates a level
    manager.process_pivot({'price': 100.0, 'type': 'high'}, current_bar=20, current_atr=atr, timeframe='5m')
    assert len(manager.active_levels) == 1
    assert manager.active_levels[0].price == 100.0
    assert '5m' in manager.active_levels[0].tags
    
    # Second pivot within 0.08 * ATR (0.8) -> should merge
    manager.process_pivot({'price': 100.5, 'type': 'high'}, current_bar=40, current_atr=atr, timeframe='1h')
    assert len(manager.active_levels) == 1  # Merged
    assert manager.active_levels[0].price == 100.0  # Keeps original price
    assert '1h' in manager.active_levels[0].tags
    assert '5m' in manager.active_levels[0].tags

def test_level_manager_quality_and_temporal():
    manager = LevelManager()
    atr = 10.0
    
    # Level 1
    manager.process_pivot({'price': 100.0, 'type': 'high'}, current_bar=20, current_atr=atr, timeframe='5m')
    
    # Temporal failure (less than 18 bars later)
    manager.process_pivot({'price': 150.0, 'type': 'high'}, current_bar=30, current_atr=atr, timeframe='5m')
    assert len(manager.active_levels) == 1  # Rejected
    
    # Quality check failure (between 0.08 and 0.70 ATR distance) -> (0.8 to 7.0)
    # distance is 5.0
    manager.process_pivot({'price': 105.0, 'type': 'high'}, current_bar=40, current_atr=atr, timeframe='5m')
    assert len(manager.active_levels) == 1  # Rejected
    
    # Valid new level
    manager.process_pivot({'price': 120.0, 'type': 'high'}, current_bar=40, current_atr=atr, timeframe='5m')
    assert len(manager.active_levels) == 2  # Accepted

def test_sweep_detector():
    manager = LevelManager()
    sweep = SweepDetector(manager)
    
    # Create levels manually
    manager.process_pivot({'price': 100.0, 'type': 'high'}, current_bar=20, current_atr=10, timeframe='5m')
    manager.process_pivot({'price': 50.0, 'type': 'low'}, current_bar=40, current_atr=10, timeframe='5m')
    
    events = []
    event_bus.subscribe("LEVEL_SWEPT", lambda data: events.append(data))
    
    # Price stays between 60 and 90, no sweeps
    sweep.check_sweeps({'h': 90, 'l': 60}, current_bar=50)
    assert len(manager.active_levels) == 2
    assert len(events) == 0
    
    # Price sweeps the high level
    sweep.check_sweeps({'h': 101, 'l': 60}, current_bar=51)
    assert len(manager.active_levels) == 1
    assert manager.active_levels[0].price == 50.0
    assert len(events) == 1
    assert events[0]['price'] == 100.0
    assert events[0]['type'] == 'high'
    assert len(sweep.recent_sweeps) == 1
    
    # Advance time to test sweep cleanup (12 bars)
    sweep.check_sweeps({'h': 90, 'l': 60}, current_bar=65) # 65 - 51 = 14 > 12
    assert len(sweep.recent_sweeps) == 0

def test_liquidity_detector_e2e():
    detector = LiquidityDetector()
    
    for i in range(1, 20):
        detector._handle_5m_candle({'h': 100 + (i % 5), 'l': 90})
        
    # Verify features format
    features = detector.get_features(current_price=100.0)
    assert 'lvl_dist_above_5m' in features
    assert 'lvl_active_count' in features
    assert 'lvl_recent_sweep_count' in features
    assert 'lvl_nearest_confluence' in features
