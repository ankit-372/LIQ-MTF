"""
Shared Event Bus Module
Provides a central publish-subscribe mechanism for event-driven coordination.
"""
from src.core import event_bus

class EventBus:
    @classmethod
    def subscribe(cls, event_type: str, callback):
        """Registers a callback function for a specific event type."""
        event_bus.subscribe(event_type, callback)

    @classmethod
    def publish(cls, event_type: str, data: dict):
        """Publishes event data to all subscribed handlers of the event type."""
        event_bus.publish(event_type, data)

    @classmethod
    def clear(cls):
        """Clears all event subscribers, mostly for tests."""
        event_bus.clear_subscribers()
