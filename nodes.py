# Omer Dekel

from dataclasses import dataclass, field
from typing import TypeVar, Generic

T = TypeVar('T')

# Represents a node in the search graph.
@dataclass
class TreeSearchNode(Generic[T]):
    state: T
    depth: int
    cost: float
    parent: 'TreeSearchNode[T] | None'


# Wraps a node and adds priority for strategies that use priority queues (e.g. best first strategies).
@dataclass(order=True)
class PrioritizedTreeSearchNode(Generic[T]):
    priority: float
    node: TreeSearchNode[T] = field(compare=False)
