import argparse
import sys
from typing import Callable
from typing import Sequence

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

# Plane elasticity: 2 degrees of freedom per node
dof_per_node = 2

# Gauss quadrature points
tri_gauss_pts = [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]]
tri_gauss_wts = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]

lin_gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
lin_gauss_wts = [1.0, 1.0]

tri_edges = [[0, 1], [1, 2], [2, 0]]
dof_per_node = 2

X, Y = 0, 1
NEUMANN = 0
DIRICHLET = 1
ROBIN = 2
TRACTION = 3


def plane(
    p: NDArray[float],
    t: NDArray[float],
    D: NDArray[float],
    dbcs: list[tuple[int, int, float]] | None = None,
    nbcs: list[tuple[int, int, int, tuple[float, ...]]] | None = None,
    source: Callable[[float, float], NDArray[float]] = lambda x, y: np.zeros(2),
) -> tuple[NDArray[float], NDArray[float]]:
    """Solve the 2D plane stress/strain problem

    Args:
        p: nodal coordinates
        t: triangulation.  t[:, i] are the ordered node numbers of the ith element
        D: D[e] is the 3x3 plane stress elastic stiffness for the eth element
        dbcs: Dirichlet BCs.  dbcs[i] = [int, int, float] is the
            node number, local degree of freedom, and prescribed displacment of the ith BC
            0 based indexing.  local degrees of freedom range from 0 to number of dimensions - 1
        nbcs: Neumann BCs.  nbcs[i] = [int, int, int, tuple] is the
            element number, edge number, BC type, and value of the ith BC
            The BC type should be one of TRACTION or ROBIN
            If BC type is ROBIN, then value = (H, t0) where H is a 2x2 matrix
            If BC type is TRACTION, then value = (tx, ty)
        source: Body force.  b = s(x, y) is an array (bx, by) at the point (x, y)

    Returns:
        d: nodal displacements returned with shape (nnode, 2)
        r: nodal reactions (including Neumann terms) returned with shape (nnode, 2)

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
            fe += w * Je * np.dot(Pe, source(x, y))

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
            if type == ROBIN:
                H, t0 = value
                for w, xi in zip(lin_gauss_wts, lin_gauss_pts):
                    x = 0.5 * (1.0 - xi) * x1 + 0.5 * (1 + xi) * x2
                    y = 0.5 * (1.0 - xi) * y1 + 0.5 * (1 + xi) * y2
                    J = le / 2.0
                    Ne = shape(pe, x, y)
                    Pe = pmatrix(Ne)[nft]
                    fe[nft] += w * J * np.dot(Pe, np.dot(H, t0))
                    ke[np.ix_(nft, nft)] += w * J * np.dot(np.dot(Pe, H), Pe.T)
            elif type == TRACTION:
                tx, ty = value
                for w, xi in zip(lin_gauss_wts, lin_gauss_pts):
                    x = 0.5 * (1.0 - xi) * x1 + 0.5 * (1 + xi) * x2
                    y = 0.5 * (1.0 - xi) * y1 + 0.5 * (1 + xi) * y2
                    J = le / 2.0
                    Ne = shape(pe, x, y)
                    Pe = pmatrix(Ne)[nft]
                    fe[nft] += w * J * np.dot(Pe, [tx, ty])

        nft = dofmap(nodes)
        K[np.ix_(nft, nft)] += ke
        F[nft] += fe

    # Enforce dirichlet boundary conditions
    Kbc = K.copy()
    Fbc = F.copy()
    for node, ldof, u in dbcs:
        dof = node * dof_per_node + ldof
        Fbc -= Kbc[:, dof] * u
        Kbc[dof, :] = 0.0
        Kbc[:, dof] = 0.0
        Kbc[dof, dof] = 1.0
        Fbc[dof] = u

    # Solve for nodal displacements and reactions
    d = np.linalg.solve(Kbc, Fbc)
    r = np.dot(K, d) - F

    return d.reshape((nnode, -1)), r.reshape((nnode, -1))


def dofmap(nodes: list[int]) -> list[int]:
    return [dof_per_node * n + d for n in nodes for d in range(dof_per_node)]


def area(p: NDArray[float]) -> float:
    """Find the area of the triangle subtended by points p

    Args:
        p: p[:,0] are the x points, p[:,1] are the y points

    """
    xp, yp = p[:, 0], p[:, 1]
    d = xp[0] * (yp[1] - yp[2]) + xp[1] * (yp[2] - yp[0]) + xp[2] * (yp[0] - yp[1])
    assert d > 0
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
    P = np.zeros((6, 2))
    P[0::2, 0] = N
    P[1::2, 1] = N
    return P


def shapegrad(p: NDArray[float]) -> NDArray[float]:
    A = area(p)
    xp, yp = p[:, 0], p[:, 1]
    dN = np.zeros((2, 3))
    dN[0, :] = [yp[1] - yp[2], yp[2] - yp[0], yp[0] - yp[1]]
    dN[1, :] = [xp[2] - xp[1], xp[0] - xp[2], xp[1] - xp[0]]
    return dN / 2.0 / A


def bmatrix(dN: NDArray[float]) -> NDArray[float]:
    B = np.zeros((3, 6))
    B[0, 0::2] = dN[0]
    B[1, 1::2] = dN[1]
    B[2, 0::2] = dN[1]
    B[2, 1::2] = dN[0]
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


def tplot(p: NDArray[float], t: NDArray[int], z: NDArray[float]) -> None:
    """Make a 2D contour plot

    Args:
      p: mesh point coordinates (n, 2)
      t: mesh connectivity (triangulation) (N, 3)
      z: array of points to plot (n)

    """
    triang = tri.Triangulation(p[:, 0], p[:, 1], t)
    plt.figure(figsize=(7, 5))
    countour = plt.tricontourf(triang, z, levels=50, cmap="turbo")
    plt.triplot(triang, color="k", linewidth=0.3)
    plt.colorbar(countour, label=None)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Temperature field")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.cla()
    plt.close("all")


def indices(a: Sequence[int], values: Sequence[int]) -> list[int]:
    return [i for i, n in enumerate(a) if n in values]


def find_edge(conn: Sequence[int], nodes: Sequence[int]) -> int:
    """Given ordered nodes ``conn`` that define a triangles connectivity, find the edge defined by
    edge nodes ``nodes``"""
    li, lj = indices(conn, nodes)
    for edge, (a, b) in enumerate(tri_edges):
        if {a, b} == {li, lj}:
            return edge
    raise ValueError(f"Unable to determine edge for nodes {nodes} in triangle {conn}")


def isclose(a, b, rtol: float = 0.0001, atol: float = 1e-8) -> bool:
    return abs(a - b) <= (atol + rtol * abs(b))


def exercise(esize: float = 0.05):
    p, t = plate_with_hole(esize=esize)

    # Generate Dirichlet BCs: fixed nodes on top
    dbcs: list[tuple[int, int, float]] = []
    for node in np.where(np.abs(p[:, 1] - 1.0) < 1e-6)[0]:
        dbcs.append((node, X, 0.0))
        dbcs.append((node, Y, 0.0))

    # Generate Neumann BCs: fix top edge
    nbcs: list[tuple[int, int, int, tuple[float, ...]]] = []
    for e, nodes in enumerate(t):
        jlo = [node for node in nodes if isclose(p[node, 1], -1.0)]
        if len(jlo) == 2:
            # 2 nodes at Y=1 indicate this is an edge
            edge = find_edge(nodes, jlo)
            nbcs.append((e, edge, TRACTION, (400.0e3, -300.0e3)))

    E = 30.0e9
    nu = 0.3
    d = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    D = np.zeros((t.shape[0], 3, 3))
    D[:] = d
    rho = 2400.0
    g = 9.81
    s = lambda x, y: np.array([0.0, -rho * g])
    u, r = plane(p, t, D, source=s, nbcs=nbcs, dbcs=dbcs)
    scale = 0.25 / np.max(np.abs(u))
    U = np.linalg.norm(u, axis=1)
    print(np.amax(U))
    tplot(p + scale * u, t, U)


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
    p.add_argument("-s", type=float, default=0.05, help="Element size [default: %(default)s]")
    args = p.parse_args()
    exercise(esize=args.s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
