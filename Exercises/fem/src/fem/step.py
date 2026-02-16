from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from .collections import Solution
from .solver import NonlinearNewtonSolver
from .typing import DLoadT
from .typing import DSLoadT
from .typing import RLoadT

if TYPE_CHECKING:
    from .model import Model


class Step(ABC):
    def __init__(
        self,
        *,
        name: str,
        period: float = 1.0,
        parent: "Step | None" = None,
        dbcs: list[tuple[int, int, float]] | None = None,
        nbcs: list[tuple[int, int, float]] | None = None,
        dsloads: DSLoadT | None = None,
        dloads: DLoadT | None = None,
        rloads: RLoadT | None = None,
        equations: list[list[int | float]] | None = None,
    ) -> None:
        self.name = name
        self.period = period
        self.dbcs: list[tuple[int, int, float]] = dbcs or []
        self.nbcs: list[tuple[int, int, float]] = nbcs or []
        self.dsloads: DSLoadT = dsloads or {}
        self.dloads: DLoadT = dloads or {}
        self.rloads: RLoadT = rloads or {}
        self.equations: list[list[int | float]] = equations or []

        self.number: int = 1
        self.start: float = 0.0
        self._solution: Solution | None = None

    @abstractmethod
    def initialize(self, parent: "Step | None") -> None: ...

    @abstractmethod
    def solve(self, model: "Model") -> Solution: ...

    @abstractmethod
    def finalize(self, solution: Solution) -> None: ...

    @property
    def solution(self) -> Solution:
        if self._solution is None:
            raise RuntimeError("Solution has not been computed")
        return self._solution


class StaticStep(Step):
    def initialize(self, parent: Step | None) -> None:
        if parent is None:
            return
        self.number = parent.number + 1
        self.start = parent.start + parent.period

    def solve(self, model: "Model") -> Solution:
        solver = NonlinearNewtonSolver()
        return solver(model, self)

    def finalize(self, solution: Solution) -> None:
        self._solution = solution
