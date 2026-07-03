"""
Shared Event Bus Module
Provides a central publish-subscribe mechanism for event-driven coordination.
"""

class EventBus:
    _listeners = {}

    @classmethod
    def subscribe(cls, event_type: str, callback):
        """Registers a callback function for a specific event type."""
        if event_type not in cls._listeners:
            cls._listeners[event_type] = []
        cls._listeners[event_type].append(callback)

    @classmethod
    def publish(cls, event_type: str, data: dict):
        """Publishes event data to all subscribed handlers of the event type."""
        if event_type in cls._listeners:
            for callback in cls._listeners[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"[EventBus Error] Handled callback for '{event_type}' raised exception: {e}")
