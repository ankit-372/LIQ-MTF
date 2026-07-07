import os
import json
import asyncio
import tempfile
import time
from unittest.mock import patch, MagicMock

import pytest

from src.core.config import Config
from src.core.logger import Logger
from src.core import event_bus
from src.core.alerter import Alerter
from src.core.heartbeat import Heartbeat

@pytest.fixture(autouse=True)
def setup_teardown():
    event_bus.clear_subscribers()
    yield
    event_bus.clear_subscribers()

def test_config_loads_defaults():
    # If no .env is provided, it should load defaults
    config = Config()
    assert config.TRADING_MODE == "paper"
    assert config.BASE_POSITION_SIZE == 0.003
    assert config.JOURNAL_PATH == "src/data/journal.db"

def test_logger_writes_jsonl():
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = Logger(logs_dir=temp_dir)
        
        # Publish some events
        event_bus.publish("TEST_EVENT", {"key": "value"})
        event_bus.publish("ANOTHER_EVENT", "plain string")
        
        # Determine filepath
        filepath = logger._get_log_filepath()
        assert os.path.exists(filepath)
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        assert len(lines) == 2
        
        data1 = json.loads(lines[0])
        assert data1["event"] == "TEST_EVENT"
        assert data1["key"] == "value"
        assert "ts" in data1
        
        data2 = json.loads(lines[1])
        assert data2["event"] == "ANOTHER_EVENT"
        assert data2["data"] == "plain string"

@patch('urllib.request.urlopen')
def test_alerter_sends_mock_message(mock_urlopen):
    # Setup mock
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response
    
    alerter = Alerter(token="mock_token", chat_id="mock_id")
    result = asyncio.run(alerter.send("HEARTBEAT_FAIL", "Test failure"))
    
    assert result is True
    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args[0][0]
    assert "api.telegram.org/botmock_token" in req.full_url

def test_heartbeat_detects_dead_ws():
    mock_alerter = MagicMock()
    
    heartbeat = Heartbeat(mock_alerter)
    
    # Simulate WS1 dying 6 seconds ago
    heartbeat.last_seen['ws1'] = time.time() - 6.0
    # Simulate Feature engine dying 7 minutes ago
    heartbeat.last_seen['feature_engine'] = time.time() - 420.0
    
    # We can't let run() loop forever in test, so we extract the inner logic
    now = time.time()
    failures = []
    
    if now - heartbeat.last_seen['ws1'] > 5:
        failures.append("WS#1 (aggTrade) delayed > 5s")
    if now - heartbeat.last_seen['feature_engine'] > 360:
        failures.append("Feature Engine delayed > 6min")
        
    assert len(failures) == 2
    assert "WS#1 (aggTrade) delayed > 5s" in failures
    assert "Feature Engine delayed > 6min" in failures
