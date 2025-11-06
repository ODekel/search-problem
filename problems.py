# Omer Dekel

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Tuple, Iterable

T = TypeVar('T')


# Represents a single step towards the solution of a problem.
@dataclass(frozen=True)
class Step(ABC):
    cost: float
    action: Any


# A step in a problem where the cost of all actions is the same.
# In this case the cost is 1, but if all the costs are the same the specific value doesn't matter.
@dataclass(frozen=True)
class UniformCostStep(Step):
    def __init__(self, action: Any):
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "cost", 1)


# Represents a problem to be solved.
# A problem can be solved if it has a start state, knows for every state if it is a "winning" state and
# can return all the states accessible from a current state.
class Problem(Generic[T], ABC):
    def __init__(self, start_state: T):
        self._start_state = start_state

    @property
    def start_state(self) -> T:
        return self._start_state

    @abstractmethod
    def is_goal(self, state: T) -> bool:
        pass

    @abstractmethod
    def get_successors(self, state: T) -> Iterable[Tuple[Step, T]]:
        pass


# A helper class for all problems where the goal is a specific state.
class TargetGoalProblem(Generic[T], Problem[T], ABC):
    def __init__(self, start_state: T, target_state: T):
        super().__init__(start_state)
        self._target_state = target_state

    @property
    def target_state(self) -> T:
        return self._target_state

    def is_goal(self, state: T) -> bool:
        return state == self._target_state
