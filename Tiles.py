# Omer Dekel
# This file includes everything that is coupled to the 8-Tiles i.e. isn't an abstraction or a general algorithm.

import json
import operator
import sys
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Iterable, Tuple, Optional

import numpy as np
import numpy.typing as npt

from nodes import TreeSearchNode
from problems import UniformCostStep, TargetGoalProblem
from search import tree_search, tree_search_early_goal_test
from strategies import UniformCostStrategy, BreadthFirstSearchStrategy, AStarStrategy


TileState = npt.NDArray[np.uint8]


# Defines all possible actions.
# The values are regular python tuple instead of numpy arrays because numpy arrays aren't comparable and thus can't
# be used as enum values.
class TilesAction(Enum):
    Left = (0, -1)
    Right = (0, 1)
    Up = (-1, 0)
    Down = (1, 0)


# Defines a single step in the 8-Tiles game.
@dataclass(frozen=True)
class TilesStep(UniformCostStep):
    action: TilesAction
    
    def __init__(self, action: TilesAction):
        super().__init__(action)


# An implementation of the abstract Problem class for the 8-Tiles game.
class TilesProblem(TargetGoalProblem[TileState]):
    def __init__(self, start_state: TileState, target_state: TileState):
        if start_state.shape != target_state.shape:
            raise ValueError("Start state and target state must have the same shape.")

        super().__init__(start_state, target_state)

    @property
    def height(self) -> int:
        return self._start_state.shape[0]

    @property
    def width(self) -> int:
        return self._start_state.shape[1]

    # Win state is when the board state is the same as the target state.
    def is_goal(self, state: TileState) -> bool:
        return np.array_equal(state, self._target_state)

    # Successors are all the moves the blank tile can make.
    # All action in the enum above that do not cause the blank tile to go out of bounds.
    # The tile that is where the blank tile "wants" to move will replace it in its tile.
    def get_successors(self, state: TileState) -> Iterable[Tuple[TilesStep, TileState]]:
        successors = []
        empty = np.argwhere(state == 0)[0]
        for action in TilesAction:
            new_empty = empty + action.value
            if 0 <= new_empty[0] < self._start_state.shape[0] and 0 <= new_empty[1] < self._start_state.shape[1]:
                new_state = state.copy()
                new_state[empty[0], empty[1]], new_state[new_empty[0], new_empty[1]] = (
                    new_state[new_empty[0], new_empty[1]],
                    new_state[empty[0], empty[1]])
                successors.append((TilesStep(action), new_state))
        return successors


# Heuristic that counts the number of misplaced tiles (minus the blank tile).
class MisplacedTilesHeuristic:
    def __init__(self, problem: TilesProblem):
        self._problem = problem

    def evaluate(self, state: TileState) -> float:
        return max(np.sum(state != self._problem.target_state) - 1, 0)


# Heuristic that sums the manhattan distance of each tile (ignoring the blank tile).
class ManhattanDistanceHeuristic:
    def __init__(self, problem: TilesProblem):
        self._problem = problem
        self._target_positions = [0] * (problem.height * problem.width)
        for h in range(0, problem.height):
            for w in range(0, problem.width):
                value = problem.target_state[h, w]
                self._target_positions[value] = np.array([h, w], dtype=np.uint8)

    def evaluate(self, state: TileState) -> float:
        distance = 0
        for h in range(0, self._problem.height):
            for w in range(0, self._problem.width):
                value = state[h, w]
                if value == 0:
                    continue
                target_pos = self._target_positions[value]
                # Add the difference in positions to distance.
                distance += abs(h - int(target_pos[0])) + abs(w - int(target_pos[1]))
        return distance


# Heuristic that uses the manhattan distance heuristic and adds a penalty for each reversed pair of tiles.
# More information in the README.
class ManhattanDistanceWithReversalPenaltyHeuristic(ManhattanDistanceHeuristic):
    def evaluate(self, state: TileState) -> float:
        distance = super().evaluate(state)

        for h in range(0, self._problem.height):
            for w in range(0, self._problem.width):
                value = state[h, w]
                if value == 0:
                    continue

                loc = np.array([h, w], dtype=np.uint8)
                neighbor = 0
                # Every tile check only the tiles left of and above it, so every pair is checked once.
                # Find out who is the neighbor in the tile where the current value is in the target, if it is
                # right next to its current location. Then check if the neighbor's target tile is the current location
                # of the current value. If it is than we have found a reversed pair.
                if np.array_equal(loc + TilesAction.Left.value, self._target_positions[value]):
                    neighbor = state[h, w - 1]
                elif np.array_equal(loc + TilesAction.Up.value, self._target_positions[value]):
                    neighbor = state[h - 1, w]

                if neighbor == 0:
                    continue

                if np.array_equal(self._target_positions[neighbor], loc):
                    distance += 2

        return distance


def print_solution(algo_name: str, solution: Optional[TreeSearchNode[TileState]], expanded: int) -> None:
    print("Algorithm:", algo_name)
    if solution is not None:
        steps = []
        node = solution
        while node.parent is not None:
            diff = np.subtract(node.state, node.parent.state, dtype=np.int8)
            steps.append(diff[diff > 0][0])
            node = node.parent
        print("Path:", ' '.join((str(step) for step in reversed(steps))))
        print("Length:", solution.depth)
    else:
        print("No solution found")
    print("Expanded:", expanded)
    print()


with open('config.json', 'r') as f:
    target = np.array(json.load(f)['target'], dtype=np.uint8)
flat_start = [int(cell) for cell in sys.argv[1:]]
if len(flat_start) != reduce(operator.mul, target.shape):
    exit("Target and start must match")
start = np.array(flat_start, dtype=np.uint8).reshape(target.shape)
prob = TilesProblem(
    start_state=start,
    target_state=target
)

# Regular BFS search
print_solution("BFS", *tree_search_early_goal_test(prob, BreadthFirstSearchStrategy()))
# Uniform cost search. In this case, really just a worse version of BFS, not using the early goal test and
# a priority queue instead of a regular queue.
print_solution("UCS", *tree_search(prob, UniformCostStrategy()))
# A* searches using different heuristics.
print_solution("A* (Misplaced Tiles)", *tree_search(prob, AStarStrategy(MisplacedTilesHeuristic(prob).evaluate)))
print_solution("A* (Manhattan Distance)", *tree_search(prob, AStarStrategy(ManhattanDistanceHeuristic(prob).evaluate)))
print_solution("A* (Manhattan Distance with Reversal Penalty)", *tree_search(prob, AStarStrategy(ManhattanDistanceWithReversalPenaltyHeuristic(prob).evaluate)))
