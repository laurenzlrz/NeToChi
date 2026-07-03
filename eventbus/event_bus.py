"""
This module implements an event bus system, which I wanted to implement at some point, so I thought I just start
with it during this project.
"""

from typing import Callable, List, Type, Any
from dataclasses import dataclass
from typing import Union, assert_never

from typing_extensions import Dict


class TypeToken[T]:
    """
    A generic type token class that can be used to represent a type at runtime.
    This is useful for scenarios where you need to work with types in a generic way.
    """
    def __init__(self, type_: Type[T]) -> None:
        self.type_ = type_
        

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TypeToken):
            return self.type_ == other.type_
        return False

    def __hash__(self) -> int:
        return hash(self.type_)


class BaseEvent:

    def __init__(self, type_token: TypeToken[Any] = None) -> None:
        self._type_token = type_token

    @property
    def type_token(self) -> TypeToken[Any] | None:
        return self._type_token


class EventBus[E: BaseEvent]:
    def __init__(self) -> None:
        self._pattern_matching_subscribers: List[Callable[[E], None]] = []
        self._map_subscribers: Dict[TypeToken[Any], Any] = {}

    def subscribe_pattern_matched(self, subscriber: Callable[[E], None]) -> None:
        """Interface for subscribing to events. Subscribers should be callable functions that accept an event of type E."""
        self._pattern_matching_subscribers.append(subscriber)

    def subscribe_class(self, event_class: TypeToken[E], subscriber: Callable[[E], Any]) -> None:
        """Subscribe to events of a specific class."""
        if event_class not in self._map_subscribers:
            self._map_subscribers[event_class] = []

        self._map_subscribers[event_class].append(subscriber)

    def publish(self, event: E) -> None:
        """Publish an event to all subscribers."""
        for subscriber in self._pattern_matching_subscribers:
            subscriber(event)

        if event.type_token is not None:
            if event.type_token in self._map_subscribers:
                for subscriber in self._map_subscribers[event.type_token]:
                    subscriber(event)

        for cls in type(event).__mro__:
            type_token = TypeToken(cls)
            if type_token in self._map_subscribers:
                for subscriber in self._map_subscribers[type_token]:
                    subscriber(event)
