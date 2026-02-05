import argparse
import sys
from functools import reduce
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from distmeshpy import distmesh2d
from distmeshpy.utils import dcircle
from distmeshpy.utils import ddiff
from distmeshpy.utils import drectangle
from distmeshpy.utils import huniform
from numpy.typing import NDArray

np.set_printoptions(precision=2)

# Heat transfer: 1 degree of freedom per node
dof_per_node = 1

# Gauss quadrature points
tri_gauss_pts = [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]]
tri_gauss_wts = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]

lin_gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
lin_gauss_wts = [1.0, 1.0]

tri_edges = [[0, 1], [1, 2], [2, 0]]

NEUMANN = 0
DIRICHLET = 1
CONVECTION = 2
CONDUCTION = 3


def heat2d(
    p: NDArray[float],
    t: NDArray[float],
    D: NDArray[float],
    dbcs: list[tuple[int, float]] | None = None,
    nbcs: list[tuple[int, int, int, tuple[float, ...]]] | None = None,
    source: Callable[[float, float], float] = lambda x, y: 0.0,
) -> tuple[NDArray[float], NDArray[float]]:
    """Solve the 2D heat problem

    Args:
        p: nodal coordinates
        t: triangulation.  t[:, i] are the ordered node numbers of the ith element
        D: D[e] is the scalar thermal conductivity of the eth element
        dbcs: Dirichlet BCs.  dbcs[i] = [int, float] is the
            node number and prescribed temperature of the ith BC
        nbcs: Neumann BCs.  nbcs[i] = [int, int, int, tuple] is the
            element number, edge number, BC type, and value of the ith BC
            The BC type should be one of CONDUCTION or CONVECTION
            If BC type is CONVECTION, then value = (h, t0) (convection coefficient and far field temp)
            If BC type is CONDUCTION, then value = (q,) (conduction on edge)
        source: Heat generation (s(x, y))

    Returns:
        temp: nodal temperatures
        r: nodal reactions (including Neumann terms)

    """
    nnode = p.shape[0]
    ndof = dof_per_node * nnode
    node_per_elem = t.shape[1]
    dof_per_elem = node_per_elem * dof_per_node

    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    dbcs = dbcs or []
    nbciter: NeumannBoundaryIterator = NeumannBoundaryIterator(nbcs or [])

    for e, nodes in enumerate(t):
        pe = p[nodes]
        dN = shapegrad(pe)
        Be = bmatrix(dN)
        Ae = area(pe)
        Je = 2.0 * Ae

        # Element stiffness matrix
        ke = Ae * np.dot(np.dot(Be.T, D[e]), Be)

        # Element body source array
        # The equivalent nodal fluxes for the spatially-varying source is found by integrating
        # {Ne(x,y)} * s(x,y), over the element domain using 3-pt Gauss quadrature.
        fe = np.zeros(dof_per_elem)
        xp, yp = pe[:, 0], pe[:, 1]
        for w, tc in zip(tri_gauss_wts, tri_gauss_pts):
            # NOTE: the triangular coordinate `tc` is also a shape function
            x = xp[0] * tc[0] + xp[1] * tc[1] + xp[2] * (1.0 - tc[0] - tc[1])
            y = yp[0] * tc[0] + yp[1] * tc[1] + yp[2] * (1.0 - tc[0] - tc[1])
            Ne = shape(pe, x, y)
            Pe = pmatrix(Ne)
            fe += w * Je * np.dot(Pe, [source(x, y)])

        # Element boundary source arrays
        # The equivalent nodal fluxes for the Neumann BCs is found by integrating
        # {Ne(x,y)} * q, along the effected edge using 2-pt Gauss quadrature.
        for nbc in nbciter.get(e):
            edge, type, value = nbc
            edge_nodes = tri_edges[edge]  # local node freedom table
            nft = dofmap(edge_nodes)
            x1, y1 = pe[edge_nodes[0]]
            x2, y2 = pe[edge_nodes[1]]
            le = np.hypot(x2 - x1, y2 - y1)
            if type == CONVECTION:
                h, t0 = value
                for w, xi in zip(lin_gauss_wts, lin_gauss_pts):
                    x = 0.5 * (1.0 - xi) * x1 + 0.5 * (1 + xi) * x2
                    y = 0.5 * (1.0 - xi) * y1 + 0.5 * (1 + xi) * y2
                    J = le / 2.0
                    Ne = shape(pe, x, y)
                    Pe = pmatrix(Ne)[nft]
                    fe[nft] += w * J * np.dot(Pe, [h * t0])
                    ke[np.ix_(nft, nft)] += w * J * np.dot(np.dot(Pe, h), Pe.T)
            elif type == CONDUCTION:
                qb, *_ = value
                for w, xi in zip(lin_gauss_wts, lin_gauss_pts):
                    x = 0.5 * (1.0 - xi) * x1 + 0.5 * (1 + xi) * x2
                    y = 0.5 * (1.0 - xi) * y1 + 0.5 * (1 + xi) * y2
                    J = le / 2.0
                    Ne = shape(pe, x, y)
                    Pe = pmatrix(Ne)[nft]
                    fe[nft] += w * J * np.dot(Pe, [qb])

        nft = dofmap(nodes)
        K[np.ix_(nft, nft)] += ke
        F[nft] += fe

    # Enforce dirichlet boundary conditions
    Kbc = K.copy()
    Fbc = F.copy()
    for node, T in dbcs:
        dof = node * dof_per_node + 0
        Kbc[dof, :] = 0.0
        Kbc[dof, dof] = 1.0
        Fbc[dof] = T

    # Solve for nodal temperature and reactions
    temp = np.linalg.solve(Kbc, Fbc)
    r = np.dot(K, temp) - F

    return temp, r


def dofmap(nodes: list[int]) -> list[int]:
    return [dof_per_node * n + d for n in nodes for d in range(dof_per_node)]


def area(p: NDArray[float]) -> float:
    """Find the area of the triangle subtended by points p

    Args:
        p: p[:,0] are the x points, p[:,1] are the y points

    """
    xp, yp = p[:, 0], p[:, 1]
    d = xp[0] * (yp[1] - yp[2]) + xp[1] * (yp[2] - yp[0]) + xp[2] * (yp[0] - yp[1])
    return d / 2.0


def shape(p: NDArray[float], x: float, y: float) -> NDArray[float]:
    A = area(p)
    xp, yp = p[:, 0], p[:, 1]
    N = np.zeros(3)
    N[0] = xp[1] * yp[2] - xp[2] * yp[1] + (yp[1] - yp[2]) * x + (xp[2] - xp[1]) * y
    N[1] = xp[2] * yp[0] - xp[0] * yp[2] + (yp[2] - yp[0]) * x + (xp[0] - xp[2]) * y
    N[2] = xp[0] * yp[1] - xp[1] * yp[0] + (yp[0] - yp[1]) * x + (xp[1] - xp[0]) * y
    return N / 2.0 / A


def pmatrix(N: NDArray[float]) -> NDArray[float]:
    """The element P matrix (matrix of the shape functions)

    Args:
      N: shape function

    Returns:
      P: shape functions such that f = P.d for d = [d1 d2 ... dn]

    3, 1 . 1, 3

    """
    P = np.zeros((3, 1))
    P[:, 0] = N
    return P


def shapegrad(p: NDArray[float]) -> NDArray[float]:
    A = area(p)
    xp, yp = p[:, 0], p[:, 1]
    B = np.zeros((2, 3))
    B[0, :] = [yp[1] - yp[2], yp[2] - yp[0], yp[0] - yp[1]]
    B[1, :] = [xp[2] - xp[1], xp[0] - xp[2], xp[1] - xp[0]]
    return B / 2.0 / A


def bmatrix(dN: NDArray[float]) -> NDArray[float]:
    """The element B matrix (matrix of derivatives of the shape functions)

    Args:
      dN: shape function gradient

    Returns:
      B: shape function gradient transformed such that grad(d) = B.d for d = [d1 d2 ... dn]

    Note:
      For scalar-valued problems, the B matrix is dN
    """
    B = np.zeros((2, 3))
    B[0] = dN[0]
    B[1] = dN[1]
    return B


class NeumannBoundaryIterator:
    def __init__(self, nbcs: list[tuple[int, int, int, tuple[float, ...]]]) -> None:
        self.data = sorted(nbcs, key=lambda x: (x[0], x[1]))
        self.position = 0
        self.size = len(self.data)

    def get(self, e: int) -> list[tuple[int, int, tuple[float, ...]]]:
        result: list[tuple[int, int, tuple[float, ...]]] = []
        while self.position < self.size:
            eid, edge, type, value = self.data[self.position]
            if eid < e:
                self.position += 1
                continue
            if eid > e:
                break
            result.append((edge, type, value))
            self.position += 1
        return result


def tplot(
    p: NDArray[float],
    t: NDArray[int],
    z: NDArray[float],
    label: str = "Temperature",
    title: str = "FEA Solution",
) -> None:
    """Make temperature contour plot"""
    triang = tri.Triangulation(p[:, 0], p[:, 1], t)
    plt.figure(figsize=(7, 5))
    countour = plt.tricontourf(triang, z, levels=50, cmap="turbo")
    plt.triplot(triang, color="k", linewidth=0.3)
    plt.colorbar(countour, label=label)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.cla()
    plt.close("all")


def tplot3d(
    p: NDArray[float],
    t: NDArray[int],
    z: NDArray[float],
    label: str = "Temperature",
    title: str = "Temperature field",
) -> None:
    """Make temperature contour plot"""
    triang = tri.Triangulation(p[:, 0], p[:, 1], t)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_trisurf(triang, z, cmap="turbo", linewidth=.2, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(label)
    fig.colorbar(surf, ax=ax, shrink=0.6, label=label)

    plt.title(title)
    plt.show()

    plt.clf()
    plt.cla()
    plt.close("all")


def rplot(p: NDArray[float], t: NDArray[int], r: NDArray[float]) -> None:
    """Make plots of reactions on left and right edges"""
    _, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Left reaction
    ilo = [n for n, x in enumerate(p) if isclose(x[0], -1.0)]
    ylo = p[ilo, 1]
    rlo = r[ilo]
    ix = np.argsort(ylo)
    ylo = ylo[ix]
    rlo = rlo[ix]

    axs[0].plot(ylo, rlo, "o", label="LHS")
    axs[0].set_title("LHS reaction")
    axs[0].set_xlabel("y")
    axs[0].set_ylabel("Heat flux/reaction")
    axs[0].grid(True)

    # Right reaction
    ihi = [n for n, x in enumerate(p) if isclose(x[0], 1.0)]
    yhi = p[ihi, 1]
    rhi = r[ihi]
    ix = np.argsort(yhi)
    yhi = yhi[ix]
    rhi = rhi[ix]

    axs[1].plot(yhi, rhi, "o", label="RHS")
    axs[1].set_title("RHS reaction")
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("Heat flux/reaction")
    axs[1].grid(True)

    print(np.sum(rlo))
    print(np.sum(rhi))

    plt.tight_layout()
    plt.show()


def indices(a: list[int], values: list[int]) -> list[int]:
    return [i for i, n in enumerate(a) if n in values]


def isclose(a, b, rtol: float = 0.0001, atol: float = 1e-8) -> bool:
    return abs(a - b) <= (atol + rtol * abs(b))


def example(esize: float = 0.05):
    """
    Solve the 2D heat equation over a square domain

    • Bounds: x ∈ [-1, 1], y ∈ [-1, 1] with hole of radius .3 at its center.
    • Spatially varying heat source: 1000 / √(x^2 + y^2)
    • Fixed temperature on left edge: 200˚
    • Fixed temperature on right edge: 50˚
    • Heat flux along bottom edge: 2000
    • Convection along top edge: far field temperature 25˚ with convection coefficient 250

    """
    p, t = plate_with_hole(esize=esize)

    # Generate Dirichlet BCs: fixed temperature on left (200˚) and right (50˚) sides
    dbcs: list[tuple[int, float]] = []
    for node in np.where(np.abs(p[:, 0] + 1.0) < 1e-6)[0]:
        dbcs.append((node, 200.0))
    for node in np.where(np.abs(p[:, 0] - 1.0) < 1e-6)[0]:
        dbcs.append((node, 50.0))

    # Generate Neumann BCs: convection on top surface and conduction on bottom
    nbcs: list[tuple[int, int, int, tuple[float, ...]]] = []
    for e, nodes in enumerate(t):
        jhi = [node for node in nodes if isclose(p[node, 1], 1.0)]
        if len(jhi) == 2:
            li, lj = indices(nodes, jhi)
            for edge, (a, b) in enumerate(tri_edges):
                if {a, b} == {li, lj}:
                    nbcs.append((e, edge, CONVECTION, (250.0, 25.0)))
                    break
            continue
        jlo = [node for node in nodes if isclose(p[node, 1], -1.0)]
        if len(jlo) == 2:
            li, lj = indices(nodes, jlo)
            for edge, (a, b) in enumerate(tri_edges):
                if {a, b} == {li, lj}:
                    nbcs.append((e, edge, CONDUCTION, (2000.0,)))
                    break
            continue

    # Heat source
    s = lambda x, y: 1000.0 / np.sqrt(x**2 + y**2)
    D = np.ones(t.shape[0]) * 12.0
    temp, r = heat2d(p, t, D, source=s, nbcs=nbcs, dbcs=dbcs)
    tplot(p, t, temp)
    rplot(p, t, r)


def verify(esize: float = 0.05):
    """
    Solve the 2D heat equation over a square domain

    • Bounds: x ∈ [-1, 1], y ∈ [-1, 1]
    • Spatially varying heat source: None
    • Left and right edges insulated.
    • Heat flux along bottom edge: 2000
    • Convection along top edge: far field temperature 25˚ with convection coefficient 250

    Analytic solution:

    T = 33˚ along the bottom edge
    T = 366.333˚ along the top edge

    """
    p, t = uniform_plate(esize=esize)

    # enerate Neumann BCs: convection on top surface and conduction on bottom
    nbcs: list[tuple[int, int, int, tuple[float, ...]]] = []
    for e, nodes in enumerate(t):
        jhi = [node for node in nodes if isclose(p[node, 1], 1.0)]
        if len(jhi) == 2:
            li, lj = indices(nodes, jhi)
            for edge, (a, b) in enumerate(tri_edges):
                if {a, b} == {li, lj}:
                    nbcs.append((e, edge, CONVECTION, (250.0, 25.0)))
                    break
            continue
        jlo = [node for node in nodes if isclose(p[node, 1], -1.0)]
        if len(jlo) == 2:
            li, lj = indices(nodes, jlo)
            for edge, (a, b) in enumerate(tri_edges):
                if {a, b} == {li, lj}:
                    nbcs.append((e, edge, CONDUCTION, (2000.0,)))
                    break
            continue

    # No heat source
    s = lambda x, y: 0.0
    D = np.ones(t.shape[0]) * 12.0
    temp, r = heat2d(p, t, D, source=s, nbcs=nbcs)
    thi = temp[np.where(np.abs(p[:, 1] - 1.0) < 1e-6)[0]]
    assert np.allclose(thi, 33)
    tlo = temp[np.where(np.abs(p[:, 1] + 1.0) < 1e-6)[0]]
    assert np.allclose(tlo, 366.33333)
    tplot(p, t, temp)


def mms(esize: float = 0.05):
    p, t = uniform_plate(esize=esize)

    k = 12.0
    T = lambda x, y: np.cos(12 * x ** 2) * y
    s = lambda x, y: 24 * k * y * (np.sin(12 * x ** 2) + 24 * x ** 2 * np.cos(12 * x ** 2))

    # Generate Dirichlet BCs
    ilo = np.where(np.abs(p[:, 0] + 1.0) < 1e-6)[0]  # LHS
    jlo = np.where(np.abs(p[:, 1] + 1.0) < 1e-6)[0]  # Bottom
    ihi = np.where(np.abs(p[:, 0] - 1.0) < 1e-6)[0]  # RHS
    jhi = np.where(np.abs(p[:, 1] - 1.0) < 1e-6)[0]  # Top
    boundary_nodes = reduce(np.union1d, (ilo, jlo, ihi, jhi))
    prescribed_temperatures = T(p[boundary_nodes, 0], p[boundary_nodes, 1])
    dbcs = list(zip(boundary_nodes, prescribed_temperatures))

    D = np.ones(t.shape[0]) * k
    temp, r = heat2d(p, t, D, source=s, dbcs=dbcs)
    analytic = T(p[:, 0], p[:, 1])
    tplot(p, t, temp)
    tplot3d(p, t, temp)
    tplot3d(p, t, temp - analytic, label="$T - T_{ana}$", title="Error in FEA solution")


def plate_with_hole(esize: float) -> tuple[NDArray[float], NDArray[int]]:
    fd = lambda p: ddiff(drectangle(p, -1, 1, -1, 1), dcircle(p, 0, 0, 0.5))
    fh = lambda p: 0.05 + 0.3 * dcircle(p, 0, 0, 0.5)
    bbox = ((-1.0, 1.0), (-1.0, 1.0))
    fixed = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    p, t = distmesh2d(fd, fh, esize, bbox, fixed)
    return p, t


def uniform_plate(esize: float) -> tuple[NDArray[float], NDArray[int]]:
    fd = lambda p: drectangle(p, -1, 1, -1, 1)
    fh = lambda p: huniform(p)
    bbox = ((-1.0, 1.0), (-1.0, 1.0))
    fixed = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    p, t = distmesh2d(fd, fh, esize, bbox, fixed)
    return p, t


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?", default="example", choices=("verify", "mms", "example"))
    p.add_argument("-s", type=float, default=0.05, help="Element size [default: %(default)s]")
    args = p.parse_args()
    if args.problem == "example":
        example(esize=args.s)
    elif args.problem == "verify":
        verify(esize=args.s)
    elif args.problem == "mms":
        mms(esize=args.s)
    else:
        p.error(f"Unknown problem {args.problem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
