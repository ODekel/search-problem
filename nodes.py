# Omer Dekel

from dataclasses import dataclass, field

# Represents a node in the search graph.
@dataclass
class TreeSearchNode[T]:
    state: T
    depth: int
    cost: float
    parent: TreeSearchNode[T] | None


# Wraps a node and adds priority for strategies that use priority queues (e.g. best first strategies).
@dataclass(order=True)
class PrioritizedTreeSearchNode[T]:
    priority: float
    node: TreeSearchNode[T] = field(compare=False)
