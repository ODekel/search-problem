# Omer Dekel

from abc import ABC, abstractmethod
from queue import PriorityQueue, Queue
from typing import TypeVar, Generic, Dict, Callable, Iterable

from nodes import PrioritizedTreeSearchNode, TreeSearchNode

T = TypeVar('T')


# Helper class to handle the logic of keeping track of reached nodes for strategies.
# Uses a dictionary to track all nodes reached and their costs. The keys for the dictionary are string instead of T
# because some types (such as numpy arrays) are not hashable and so can't be used as dictionary keys.
class _ReachedStatesSearchStrategyHelper(Generic[T]):
    def __init__(self):
        self._reached: Dict[str, float] = {}

    def put_if_needed(self, node: TreeSearchNode[T], put: Callable[[TreeSearchNode[T]], None]) -> None:
        state_key = str(node.state)
        value = self._reached.get(state_key)
        if value is None or node.cost < value:
            # Does not remove the old node from the old node from the fringe because the pseudocode in the book (page 91)
            # doesn't, and it would be extremely difficult in the case of queues (and make time complexity worse).
            self._reached[state_key] = node.cost
            put(node)


# Represents a strategy for the search, strategy being the order in which nodes are searched.
# A strategy in the most basic sense is a wrapper around a data structure, where to that data structure you can add
# multiple elements (push_many because usually an array of elements is added after expanding, so it is more useful
# that way), get the next element to search and check if any more elements are available to search.
class SearchStrategy(Generic[T], ABC):
    @abstractmethod
    def push_many(self, nodes: Iterable[TreeSearchNode[T]]) -> None:
        pass

    @abstractmethod
    def pop(self) -> TreeSearchNode[T]:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass


# A breadth first strategy to search, using a simple queue.
# Nodes encountered first are searched first.
class BreadthFirstSearchStrategy(Generic[T], SearchStrategy[T]):
    def __init__(self):
        super().__init__()
        self._fringe: Queue[TreeSearchNode[T]] = Queue()
        self._reached_states_helper = _ReachedStatesSearchStrategyHelper[T]()

    def push_many(self, nodes: Iterable[TreeSearchNode[T]]) -> None:
        for node in nodes:
            self._reached_states_helper.put_if_needed(node, self._fringe.put)

    def pop(self) -> TreeSearchNode[T]:
        return self._fringe.get()

    def is_empty(self) -> bool:
        return self._fringe.empty()


# Abstract base class for best first strategies, using a priority queue.
# Implementing classes need to decide the priority of every node before using this class to push it.
class BestFirstSearchStrategy(Generic[T], SearchStrategy[T], ABC):
    def __init__(self):
        super().__init__()
        self._fringe: PriorityQueue[PrioritizedTreeSearchNode[T]] = PriorityQueue()
        self._reached_states_helper = _ReachedStatesSearchStrategyHelper[T]()

    # Pass the regular node to the helper, but push the prioritized node to the fringe.
    def _push(self, node: TreeSearchNode[T], priority: float) -> None:
        self._reached_states_helper.put_if_needed(node, lambda n: self._fringe.put(PrioritizedTreeSearchNode(priority, n)))

    def pop(self) -> TreeSearchNode[T]:
        return self._fringe.get().node

    def is_empty(self) -> bool:
        return self._fringe.empty()


# Uniform cost search (dijkstra's algorithm) strategy.
# The most basic implementation of best first search, using only the cost as the priority.
class UniformCostStrategy(Generic[T], BestFirstSearchStrategy[T]):
    def __init__(self):
        super().__init__()

    def push_many(self, nodes: Iterable[TreeSearchNode[T]]) -> None:
        for node in nodes:
            self._push(node, priority=node.cost)


# A* search strategy, using the cost of the node and a provided heuristic function to calculate the priority of a node.
class AStarStrategy(Generic[T], BestFirstSearchStrategy[T]):
    def __init__(self, heuristic: Callable[[T], float]):
        super().__init__()
        self._heuristic = heuristic

    def push_many(self, nodes: Iterable[TreeSearchNode[T]]) -> None:
        for node in nodes:
            self._push(node, priority=node.cost + self._heuristic(node.state))
