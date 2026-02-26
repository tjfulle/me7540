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

    def advance_state(self) -> None:
        self.u[0, :] = self.u[1, :]
        self.R[0, :] = self.R[1, :]

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def solve(self) -> None:
        file = ExodusFile(self)
        for i, step in enumerate(self.steps):
            step.solve(self)
            self.advance_state()
            for block in self.blocks:
                block.advance_state()
            file.update(i + 1, step)

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


class ExodusFile:
    def __init__(self, model: Model) -> None:
        fname = "_".join(model.name.split()) + ".exo"
        file = exodusii.exo_file(fname, mode="w")
        file.put_init(
            f"fem solution for {model.name}",
            num_dim=model.coords.shape[1],
            num_nodes=model.coords.shape[0],
            num_elem=model.connect.shape[0],
            num_elem_blk=len(model.blocks),
            num_node_sets=len(model.mesh.nodesets),
            num_side_sets=len(model.mesh.sidesets),
        )
        coord_names = [f"disp{'xyz'[i]}" for i in range(model.coords.shape[1])]
        file.put_coord_names(coord_names)
        file.put_coords(model.coords)

        file.put_map(model.elem_map.lid_to_gid)
        file.put_node_id_map(model.node_map.lid_to_gid)

        for i, block in enumerate(model.blocks):
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

        # write element variable truth table
        elem_vars: list[str] = []
        for block in model.blocks:
            for elem_var in block.element_variable_names():
                if elem_var not in elem_vars:
                    elem_vars.append(elem_var)
        truth_tab = np.empty((len(model.blocks), len(elem_vars)), dtype=int)
        for i, block in enumerate(model.blocks):
            for j, elem_var in enumerate(elem_vars):
                if elem_var in block.element_variable_names():
                    truth_tab[i, j] = 1
        file.put_element_variable_params(len(elem_vars))
        file.put_element_variable_names(elem_vars)
        file.put_element_variable_truth_table(truth_tab)

        nodeset_id = 1
        for name, lids in model.nodesets.items():
            gids = [model.node_map[lid] for lid in lids]
            file.put_node_set_param(nodeset_id, len(gids), 0)
            file.put_node_set_name(nodeset_id, str32(name))
            file.put_node_set_nodes(nodeset_id, gids)
            nodeset_id += 1

        sideset_id = 1
        for name, ss in model.sidesets.items():
            file.put_side_set_param(sideset_id, len(ss), 0)
            file.put_side_set_name(sideset_id, str32(name))
            gids = [model.elem_map[_[0]] for _ in ss]
            sides = [_[1] + 1 for _ in ss]
            file.put_side_set_sides(sideset_id, gids, sides)
            sideset_id += 1

        # write results variables parameters and names
        file.put_global_variable_params(1)
        file.put_global_variable_names(["time_step"])

        # Put first step
        file.put_time(1, 0.0)

        # global values
        file.put_global_variable_values(1, np.zeros(1, dtype=float))

        # nodal values
        file.put_node_variable_params(model.coords.shape[1])
        displ_names = [self.displ_name(i) for i in range(model.coords.shape[1])]
        file.put_node_variable_names(displ_names)
        for i in range(model.coords.shape[1]):
            file.put_node_variable_values(1, self.displ_name(i), model.u[0, i::2])

        # element values
        for j, block in enumerate(model.blocks):
            for name, value in block.element_variable_values():
                file.put_element_variable_values(1, j + 1, name, value)

        self.file = file
        self.model = model

        return

    def update(self, step_no: int, step: Step) -> None:
        file = self.file
        model = self.model
        file.put_time(step_no + 1, step.start + step.period)
        for i in range(model.coords.shape[1]):
            file.put_node_variable_values(step_no + 1, self.displ_name(i), model.u[0, i::2])
        for j, block in enumerate(model.blocks):
            for name, value in block.element_variable_values():
                file.put_element_variable_values(step_no + 1, j + 1, name, value)

    def displ_name(self, i: int) -> str:
        return f"displ{'xyz'[i]}"


def str32(string: str) -> str:
    return f"{string:32}"
