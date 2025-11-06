from typing import TypeVar, Tuple, Optional, Callable, Iterable

from nodes import TreeSearchNode
from problems import Problem, Step
from strategies import SearchStrategy

_create_node = lambda state, depth, cost, parent: TreeSearchNode(state, depth, cost, parent)

T = TypeVar('T')


# Tree search, implemented the same as in page 91 in the book.
# Strategy is responsible for storing the fringe, so that it can choose the data structure that best fits it.
def tree_search(problem: Problem[T], strategy: SearchStrategy[T]) -> Tuple[Optional[TreeSearchNode[T]], int]:
    expanded = 0
    node = _create_node(problem.start_state, 0, 0, None)
    strategy.push_many((node,))
    while not strategy.is_empty():
        node = strategy.pop()
        if problem.is_goal(node.state):
            return node, expanded
        children = _expand(node, problem.get_successors, _create_node)
        expanded += 1
        strategy.push_many(children)
    return None, expanded


# Tree search but with the goal test when a node is generated instead of when it is popped from the fringe.
# Some strategies (such as breadth first) can use it for a more efficient search that will expand fewer nodes.
# Implemented the same as in page 95 in the book.
def tree_search_early_goal_test(problem: Problem[T], strategy: SearchStrategy[T]) -> Tuple[Optional[TreeSearchNode[T]], int]:
    expanded = 0
    node = _create_node(problem.start_state, 0, 0, None)
    if problem.is_goal(node.state):
        return node, expanded
    strategy.push_many((node,))
    while not strategy.is_empty():
        node = strategy.pop()
        children = _expand(node, problem.get_successors, _create_node)
        expanded += 1
        for child in children:
            if problem.is_goal(child.state):
                return child, expanded
        strategy.push_many(children)
    return None, expanded


# Create a list of all successor nodes, meaning all states reachable from the current state with the extra properties
# of a node (parent, cost, etc.).
def _expand(node: TreeSearchNode[T],
            get_successors: Callable[[T], Iterable[Tuple[Step, T]]],
            create_node: Callable[[T, int, float, TreeSearchNode[T]], TreeSearchNode[T]]) -> Iterable[TreeSearchNode[T]]:
    successors = []
    for step, result in get_successors(node.state):
        successors.append(create_node(result, node.depth + 1, node.cost + step.cost, node))
    return successors
