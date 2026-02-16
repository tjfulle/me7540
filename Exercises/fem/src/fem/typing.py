from typing import Callable
from typing import Sequence

from numpy.typing import NDArray

from .collections import DistributedLoad
from .collections import DistributedSurfaceLoad
from .collections import RobinLoad

RegionSelector = Callable[[Sequence[float], bool], bool]
NodeFunction = Callable[[int, int, int, int, NDArray, list[float], float], float]

DSLoadT = dict[int, dict[int, list[tuple[int, DistributedSurfaceLoad]]]]
RLoadT = dict[int, dict[int, list[RobinLoad]]]
DLoadT = dict[int, dict[int, list[DistributedLoad]]]
# step, increment, node, x, [ti, ts], dt
