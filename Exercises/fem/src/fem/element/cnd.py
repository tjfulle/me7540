from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..material import Material
from .geom import P3
from .geom import P4
from .isop import IsoparametricElement


class CnD:
    ndir: int
    nshr: int

    @property
    def ntens(self) -> int:
        return self.ndir + self.nshr

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
    ) -> tuple[NDArray, NDArray]:
        D, s = material.eval(e, ndir=self.ndir, nshr=self.nshr)
        hsv[: self.ntens] = e
        hsv[self.ntens : 2 * self.ntens] = s
        return D, s


class CPX3(P3, CnD, IsoparametricElement):
    gauss_pts = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]], dtype=float) / 6.0
    gauss_wts = np.array([1.0, 1.0, 1.0], dtype=float) / 6.0

    edge_gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    edge_gauss_wts = np.array([1.0, 1.0], dtype=float)

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        return [
            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        ]

    def pmatrix(self, xi: NDArray) -> NDArray:
        N = self.shape(xi)
        P = np.zeros((6, 2))
        P[0::2, 0] = N
        P[1::2, 1] = N
        return P


class CPS3(CPX3):
    """Plane stress CST element"""

    ndir = 2
    nshr = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((3, 6))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[2, 0::2] = dNdx[1]
        B[2, 1::2] = dNdx[0]
        return B

    def history_variables(self) -> list[str]:
        return ["strain_xx", "strain_yy", "strain_xy", "stress_xx", "stress_yy", "stress_xy"]


class CPE3(CPX3):
    """Plane strain CST element"""

    ndir = 3
    nshr = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((4, 6))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[3, 0::2] = dNdx[1]
        B[3, 1::2] = dNdx[0]
        return B

    def history_variables(self) -> list[str]:
        return [
            "strain_xx",
            "strain_yy",
            "strain_zz",
            "strain_xy",
            "stress_xx",
            "stress_yy",
            "stress_zz",
            "stress_xy",
        ]


class CPX4(P4, CnD, IsoparametricElement):
    gauss_pts = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]) / np.sqrt(3.0)
    gauss_wts = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

    edge_gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    edge_gauss_wts = np.array([1.0, 1.0], dtype=float)

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        return [
            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        ]

    def pmatrix(self, xi: NDArray) -> NDArray:
        N = self.shape(xi)
        P = np.zeros((8, 2))
        P[0::2, 0] = N
        P[1::2, 1] = N
        return P


class CPS4(CPX4):
    """Plane stress quad element"""

    ndir = 2
    nshr = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((3, 8))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[2, 0::2] = dNdx[1]
        B[2, 1::2] = dNdx[0]
        return B

    def history_variables(self) -> list[str]:
        return ["strain_xx", "strain_yy", "strain_xy", "stress_xx", "stress_yy", "stress_xy"]


class CPE4(CPX4):
    """Plane strain quad element"""

    ndir = 3
    nshr = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((4, 8))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[3, 0::2] = dNdx[1]
        B[3, 1::2] = dNdx[0]
        return B

    def history_variables(self) -> list[str]:
        return [
            "strain_xx",
            "strain_yy",
            "strain_zz",
            "strain_xy",
            "stress_xx",
            "stress_yy",
            "stress_zz",
            "stress_xy",
        ]
