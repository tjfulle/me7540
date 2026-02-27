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
from .base import Element


class IsoparametricElement(Element):
    edges: NDArray
    ref_coords: NDArray
    gauss_pts: NDArray
    gauss_wts: NDArray
    edge_gauss_pts: NDArray
    edge_gauss_wts: NDArray

    @property
    def integration_points(self) -> Generator[tuple[Any, Any], None, None]:
        yield from zip(self.gauss_wts, self.gauss_pts)

    @property
    def edge_integration_points(self) -> Generator[tuple[Any, Any], None, None]:
        yield from zip(self.edge_gauss_wts, self.edge_gauss_pts)

    @property
    def dof_per_node(self) -> int:
        return max([sum(_) for _ in self.node_freedom_table])

    @property
    def dimensions(self) -> int:
        return sum(self.node_freedom_table[0][:3])

    @property
    def nnode(self) -> int:
        return len(self.node_freedom_table)

    @property
    def npts(self) -> int:
        return len(self.gauss_wts)

    @abstractmethod
    def shape(self, xi: NDArray) -> NDArray: ...

    @abstractmethod
    def shape_derivative(self, xi: NDArray) -> NDArray: ...

    @abstractmethod
    def pmatrix(self, xi: NDArray) -> NDArray: ...

    @abstractmethod
    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray: ...

    @abstractmethod
    def ref_edge_coords(self, edge_no: int, xi: float) -> NDArray: ...

    @abstractmethod
    def edge_tangent(self, edge_no: int, p: NDArray, xi: float) -> NDArray: ...

    @abstractmethod
    def edge_normal(self, edge_no: int, p: NDArray, xi: float) -> NDArray: ...

    @abstractmethod
    def interpolate_edge(self, edge_no: int, p: NDArray, xi: float) -> NDArray: ...

    def centroid(self, p: NDArray) -> NDArray:
        return np.asarray(p).mean(axis=0)

    def jacobian(self, p: NDArray, xi: NDArray) -> float:
        dNdxi = self.shape_derivative(xi)
        dxdxi = np.dot(dNdxi, p)
        return np.linalg.det(dxdxi)

    def shape_gradient(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdxi = self.shape_derivative(xi)
        dxdxi = np.dot(dNdxi, p)
        dNdx = np.dot(np.linalg.inv(dxdxi), dNdxi)
        return dNdx

    def interpolate(self, p: NDArray, xi: NDArray) -> NDArray:
        N = self.shape(xi)
        return np.array([np.dot(N, p[:, i]) for i in range(p.shape[1])], dtype=float)

    def edge_jacobian(self, edge_no: int, p: NDArray, xi: float) -> float:
        """
        Compute Jacobian |dx/dxi| for each 1D Gauss point along a physical edge.

        Args:
            p_edge: array of shape (n_edge_nodes, 2) with coordinates of edge nodes
        Returns:
            J: array of length ngauss with Jacobian at each Gauss point
        """
        return float(np.linalg.norm(self.edge_tangent(edge_no, p, xi)))

    @abstractmethod
    def update_state(
        self,
        material: Material,
        step: int,
        increment: int,
        time: Sequence[float],
        dt: float,
        eleno: int,
        p: NDArray,
        e: NDArray,
        de: NDArray,
        hsv: NDArray,
    ) -> tuple[NDArray, NDArray]: ...

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
    ) -> tuple[NDArray, NDArray]:
        dloads = dloads or []
        dsloads = dsloads or []
        rloads = rloads or []

        ndof = self.nnode * self.dof_per_node
        re = np.zeros(ndof)
        ke = np.zeros((ndof, ndof))

        for ipt, (w, xi) in enumerate(self.integration_points):
            # ------------------
            # Volume intetration
            # ------------------
            J = self.jacobian(p, xi)
            B = self.bmatrix(p, xi)
            P = self.pmatrix(xi)
            x = self.interpolate(p, xi)

            # --- Internal terms
            e = np.dot(B, u)
            de = np.dot(B, du) / dt
            D, s = self.update_state(
                material, step, increment, time, dt, eleno, p, e, de, pdata[ipt]
            )
            ke += w * J * np.dot(np.dot(B.T, D), B)
            re += w * J * np.dot(B.T, s)

            for dload in dloads:
                value = dload(step, increment, time, dt, eleno, ipt, x.tolist())
                re -= w * J * np.dot(P, value)

        for edge_no, dsload in dsloads:
            nodes = self.edges[edge_no]
            pd = p[nodes]
            nft = [self.dof_per_node * n + d for n in nodes for d in range(self.dof_per_node)]
            for ipt, (w, xi) in enumerate(self.edge_integration_points):
                x = self.interpolate_edge(edge_no, pd, xi)
                n = self.edge_normal(edge_no, p, xi)
                traction = dsload(step, increment, time, dt, eleno, edge_no, ipt, x.tolist(), n)
                st = self.ref_edge_coords(edge_no, xi)
                P = self.pmatrix(st)[nft]
                J = self.edge_jacobian(edge_no, p, xi)
                re[nft] -= w * J * np.dot(P, traction)

        for rload in rloads:
            nodes = self.edges[rload.edge]  # local node freedom table
            nft = [self.dof_per_node * n + d for n in nodes for d in range(self.dof_per_node)]
            H = np.asarray(rload.H)
            u0 = np.asarray(rload.u0)
            for ipt, (w, xi) in enumerate(self.edge_integration_points):
                st = self.ref_edge_coords(rload.edge, xi)
                P = self.pmatrix(st)[nft]
                J = self.edge_jacobian(rload.edge, p, xi)
                kr = w * J * np.dot(np.dot(P, H), P.T)
                fr = w * J * np.dot(P, np.dot(H, u0))
                ke[np.ix_(nft, nft)] += kr
                re[nft] += np.dot(kr, u[nft]) - fr

        return ke, re
