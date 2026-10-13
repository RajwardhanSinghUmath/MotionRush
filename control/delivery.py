"""Native input delivery facts, distinct from external game acceptance."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeliveryResult:
    requested: tuple
    completed: tuple
    error: str = ''
    uncertain: bool = False

    def __post_init__(self):
        if not isinstance(self.requested, tuple) or not isinstance(self.completed, tuple):
            raise TypeError('Delivery actions must be immutable tuples')
        if self.completed != self.requested[:len(self.completed)]:
            raise ValueError('Completed actions must be a prefix of the requested sequence')

    def __bool__(self):
        return self.completed == self.requested and not self.error and not self.uncertain


def simulated_delivery(actions):
    """Evaluator-only acknowledgement; never evidence of actual game acceptance."""
    requested = tuple(actions)
    return DeliveryResult(requested, requested)
