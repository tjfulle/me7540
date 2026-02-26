from dataclasses import dataclass
from dataclasses import field
from typing import Sequence

import exodusii
import numpy as np
from numpy.typing import NDArray

from .block import ElementBlock
from .collections import Map
from .mesh import Mesh
from .step import Step

ArrayLike = NDArray | Sequence


@dataclass
class Model:
    name: str
    mesh: Mesh
    blocks: list[ElementBlock]

    num_dof: int
    dof_map: NDArray
    block_dof_map: NDArray
    node_signatures: NDArray

    steps: list[Step] = field(default_factory=list)
    u: NDArray = field(init=False)
    R: NDArray = field(init=False)

    def __post_init__(self) -> None:
        self.u = np.zeros((2, self.num_dof), dtype=float)
        self.R = np.zeros((2, self.num_dof), dtype=float)

    @property
    def nnode(self) -> int:
        return self.mesh.coords.shape[0]

    @property
    def nelem(self) -> int:
        return self.mesh.coords.shape[0]

    @property
    def node_map(self) -> Map:
        return self.mesh.node_map

    @property
    def elem_map(self) -> Map:
        return self.mesh.elem_map

    @property
    def coords(self) -> NDArray:
        return self.mesh.coords

    @property
    def connect(self) -> NDArray:
        return self.mesh.connect

    @property
    def elemsets(self) -> dict[str, list[int]]:
        return self.mesh.elemsets

    @property
    def nodesets(self) -> dict[str, list[int]]:
        return self.mesh.nodesets

    @property
    def sidesets(self) -> dict[str, list[tuple[int, int]]]:
        return self.mesh.sidesets

    @property
    def block_elem_map(self) -> dict[int, int]:
        return self.mesh.block_elem_map

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def solve(self) -> None:
        fname = "_".join(self.name.split()) + ".exo"
        file = exodusii.exo_file(fname, mode="w")
        file.put_init(
            f"fem solution for {self.name}",
            num_dim=2,
            num_nodes=self.coords.shape[0],
            num_elem=self.connect.shape[0],
            num_elem_blk=len(self.blocks),
            num_node_sets=len(self.mesh.nodesets),
            num_side_sets=len(self.mesh.sidesets),
        )
        file.put_node_variable_params(2)
        file.put_coords(self.coords)
        file.put_node_variable_names(["displx", "disply"])
        for i, block in enumerate(self.blocks):
            file.put_element_block(
                i + 1,
                block.element.family,
                num_block_elems=block.connect.shape[0],
                num_nodes_per_elem=block.element.nnode,
                num_faces_per_elem=1,
                num_edges_per_elem=3,
            )
            file.put_element_conn(i + 1, block.connect + 1)
            file.put_element_block_name(i + 1, f"Block-{i + 1}")
            elem_vars = block.element_variable_names()
            file.put_element_variable_params(len(elem_vars))
            file.put_element_variable_names(elem_vars)
        for i, step in enumerate(self.steps):
            step.solve(self)
            self.u[0, :] = self.u[1, :]
            for block in self.blocks:
                block.advance_state()
            file.put_time(i + 1, step.start + step.period)
            disp = self.u[0].reshape((self.nnode, -1))
            file.put_node_variable_values(i + 1, "displx", disp[:, 0])
            file.put_node_variable_values(i + 1, "disply", disp[:, 1])
            for j, block in enumerate(self.blocks):
                for name, value in block.element_variable_values():
                    file.put_element_variable_values(i + 1, j + 1, name, value)

    def assemble(
        self, step: Step, increment: int, time: Sequence[float], dt: float, u: NDArray, du: NDArray
    ) -> tuple[NDArray, NDArray]:
        K = np.zeros((self.num_dof, self.num_dof), dtype=float)
        R = np.zeros(self.num_dof, dtype=float)
        for b, block in enumerate(self.blocks):
            ix = self.block_dof_map[b]
            bft = ix[ix != -1]
            kb, rb = block.assemble(
                step.number,
                increment,
                time,
                dt,
                u[bft],
                du[bft],
                dloads=step.dloads.get(b),
                dsloads=step.dsloads.get(b),
                rloads=step.rloads.get(b),
            )
            K[np.ix_(bft, bft)] += kb
            R[bft] += rb
        return K, R
