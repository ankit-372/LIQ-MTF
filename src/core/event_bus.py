"""
Event Bus for the system.
Allows any part of the system to broadcast events and any other part to listen.
"""
from typing import Callable, Dict, List, Any

# Simple pub/sub dict
_subscribers: Dict[str, List[Callable]] = {}
_global_subscribers: List[Callable] = []

def subscribe(event_name: str, callback: Callable) -> None:
    """Subscribe to an event."""
    if event_name not in _subscribers:
        _subscribers[event_name] = []
    _subscribers[event_name].append(callback)

def subscribe_all(callback: Callable) -> None:
    """Subscribe to all events (receives event_name and data)."""
    _global_subscribers.append(callback)

def publish(event_name: str, data: Any = None) -> None:
    """Publish an event to all subscribers."""
    if event_name in _subscribers:
        for callback in _subscribers[event_name]:
            callback(data)
            
    for callback in _global_subscribers:
        callback(event_name, data)

def clear_subscribers() -> None:
    """Clear all subscribers (mostly for testing)."""
    _subscribers.clear()
    _global_subscribers.clear()
