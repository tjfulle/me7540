from dataclasses import dataclass
from dataclasses import field

from numpy.typing import NDArray

from .block import TopoBlock
from .collections import Map


@dataclass
class Mesh:
    coords: NDArray
    connect: NDArray
    blocks: list[TopoBlock]
    node_map: Map
    elem_map: Map
    block_elem_map: dict[int, int]
    nodesets: dict[str, list[int]] = field(default_factory=dict)
    sidesets: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    elemsets: dict[str, list[int]] = field(default_factory=dict)
