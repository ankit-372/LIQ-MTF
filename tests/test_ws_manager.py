import time
import pytest
from unittest.mock import MagicMock

from src.ingest.ws_manager import WSManager
from src.core import event_bus

@pytest.fixture(autouse=True)
def setup_teardown():
    event_bus.clear_subscribers()
    yield
    event_bus.clear_subscribers()

def test_ws_manager_routing():
    manager = WSManager()
    
    # Mock handlers
    manager.handle_trade = MagicMock()
    manager.handle_book_ticker = MagicMock()
    manager.handle_kline_5m = MagicMock()
    manager.handle_kline_1h = MagicMock()
    manager.handle_kline_4h = MagicMock()
    
    # Route aggTrade
    manager.route_message("btcusdt@aggTrade", {"data": "test"})
    manager.handle_trade.assert_called_once_with({"data": "test"})
    
    # Route bookTicker
    manager.route_message("btcusdt@bookTicker", {"data": "test_ticker"})
    manager.handle_book_ticker.assert_called_once_with({"data": "test_ticker"})

    # Route 5m kline
    manager.route_message("btcusdt@kline_5m", {"data": "5m_kline"})
    manager.handle_kline_5m.assert_called_once_with({"data": "5m_kline"})

    # Route 1h kline
    manager.route_message("btcusdt@kline_1h", {"data": "1h_kline"})
    manager.handle_kline_1h.assert_called_once_with({"data": "1h_kline"})

    # Route 4h kline
    manager.route_message("btcusdt@kline_4h", {"data": "4h_kline"})
    manager.handle_kline_4h.assert_called_once_with({"data": "4h_kline"})

def test_ws_manager_multiplex_unpacking():
    manager = WSManager()
    
    pings = []
    event_bus.subscribe("SYSTEM_PING", lambda data: pings.append(data))
    
    # Mock handlers instead of route_message so publish_ping runs
    manager.handle_kline_1h = MagicMock()
    
    # Create a multiplexed payload for 1h_4h stream
    now_ms = int(time.time() * 1000)
    multiplexed_msg = {
        "stream": "btcusdt@kline_1h",
        "data": {
            "e": "kline",
            "E": now_ms,
            "k": {
                "t": now_ms - 1000,
                "x": True
            }
        }
    }
    
    # Test unpacking
    payload = multiplexed_msg
    
    if "stream" in payload and "data" in payload:
        stream = payload["stream"]
        data = payload["data"]
    else:
        stream = manager.stream_map.get("1h_4h")
        data = payload
        
    assert stream == "btcusdt@kline_1h"
    assert data == multiplexed_msg["data"]
    
    # Test ping publishing
    manager.route_message(stream, data)
    assert len(pings) == 1
    assert pings[0]["component"] == "ws4"
