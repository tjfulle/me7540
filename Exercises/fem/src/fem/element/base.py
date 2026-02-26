from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Generator
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..collections import DistributedLoad
from ..collections import DistributedSurfaceLoad
from ..collections import RobinLoad
from ..material import Material


class Element(ABC):
    family: str

    @property
    @abstractmethod
    def nnode(self) -> int: ...

    @property
    @abstractmethod
    def dof_per_node(self) -> int: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @property
    @abstractmethod
    def node_freedom_table(self) -> list[tuple[int, ...]]: ...

    @property
    @abstractmethod
    def npts(self) -> int: ...

    @abstractmethod
    def area(self, p: NDArray) -> float: ...

    @abstractmethod
    def centroid(self, p: NDArray) -> NDArray: ...

    @abstractmethod
    def edge_centroid(self, edge_no: int, p: NDArray) -> NDArray: ...

    @abstractmethod
    def history_variables(self) -> list[str]: ...

    @abstractmethod
    def eval(
        self,
        material: Material,
        step: int,
        increment: int,
        time: Sequence[float],
        dt: float,
        eleno: int,
        p: NDArray,
        u: NDArray,
        du: NDArray,
        pdata: NDArray,
        dloads: list[DistributedLoad] | None = None,
        dsloads: list[tuple[int, DistributedSurfaceLoad]] | None = None,
        rloads: list[RobinLoad] | None = None,
    ) -> tuple[NDArray, NDArray]: ...
