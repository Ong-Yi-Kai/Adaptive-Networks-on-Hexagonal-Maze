import numpy as np
from Lattice import Lattice
from typing import Tuple, List
from scipy.linalg import solve
import pygame


class Network:

    dt = 0.01
    r = 0.1
    mu = 1
    I_0 = 100
    def f(Q): return Q**Network.mu/(1+np.abs(Q)**Network.mu)

    def __init__(self, lattice: Lattice, sources: List[Tuple], sinks: List[Tuple]):
        """
        Initialize the network with a lattice.
        """
        self.lattice = lattice
        self.N = lattice.N
        self.M = lattice.M
        self.num_nodes = self.N * self.M

        self.sinks = sinks  # list of tuples (i,j) for sinks
        self.sources = sources  # list of tuples (i,j) for sources

        # pressure at each lattice point
        self.P = np.zeros((self.N, self.M))
        # flow rate between points
        self.Q = np.zeros((self.N, self.M, 6))
        # conductance between points (neighbor that dont exist will be zero in compute_D)
        self.D = lattice.edges.copy()  

    # enumerate the nodes row wise, let the idx be u

    def ij_to_u(self, i, j): return i * self.M + j
    def u_to_ij(self, u): return u // self.M, u % self.M

    def compute_P(self):
        """
        Solve for pressure given conductance between points that obeys Kirchhoff's 
        law, where sum of flow is 0 at each node and flow rate is proportional to 
        pressure difference.
        """

        A = np.zeros((self.num_nodes, self.num_nodes))  # Coefficient matrix
        b = np.zeros((self.num_nodes, 1))

        # set up coefficient A using capacitance
        for u in range(self.num_nodes):
            i, j = self.u_to_ij(u)
            neighbors = self.lattice.get_all_neighbors(i, j)
            for (i_p, j_p, k) in neighbors:

                v = self.ij_to_u(i_p, j_p)
                D_uv = self.D[i, j, k]
                A[u, v] = D_uv
                A[u, u] -= D_uv

        # set up B
        for (i, j) in self.sinks:
            u = self.ij_to_u(i, j)
            b[u] = Network.I_0
        for (i, j) in self.sources:
            u = self.ij_to_u(i, j)
            b[u] = -Network.I_0

        p = np.linalg.pinv(A) @ b
        self.P = p.reshape(self.N, self.M).copy()

    def compute_Q(self):
        """
        Compute flow rate Q based on current pressure values and conductance where 
        Q_uv = D * (P_v - P_u)
        """
        Q_new = np.zeros((self.N, self.M, 6))
        for u in range(self.num_nodes):
            i, j = self.u_to_ij(u)
            neighbors = self.lattice.get_all_neighbors(i, j)
            for (i_p, j_p, k) in neighbors:
                D_uv = self.D[i, j, k]
                P_u = self.P[i, j]
                P_v = self.P[i_p, j_p]
                Q_uv = D_uv * (P_v - P_u)
                Q_new[i, j, k] = Q_uv
        self.Q = Q_new.copy()

    def compute_D(self):
        """
        Compute the conductance by an euler step
        dD_uv/dt = f(|Q_uv|) - rD_uv
        """
        D_new = np.zeros((self.N, self.M, 6))
        for u in range(self.num_nodes):
            i, j = self.u_to_ij(u)
            neighbors = self.lattice.get_all_neighbors(i, j)
            for _, _, k in neighbors:
                Q_uv = self.Q[i, j, k]
                D_uv = self.D[i, j, k]
                f_Q = Network.f(np.abs(Q_uv))
                dD_uv_dt = f_Q - Network.r * D_uv
                D_uv += dD_uv_dt * Network.dt
                D_new[i, j, k] = D_uv

        self.D = D_new.copy()

    def update(self, surface):
        """
        Computes P, Q, and D and update Lattice values and edgs
        """
        self.compute_P()
        self.compute_Q()
        self.compute_D()

        # update lattice values
        self.lattice.update_vert_values(self.P)
        self.lattice.update_edge_values(self.D)

        self.lattice.draw(surface)

        # draw the sources and sinks
        for (i, j) in self.sinks:
            x = (2*j+1) * self.lattice.radius if i % 2 == 0 else \
                (2 * j+2) * self.lattice.radius
            y = (2+3*i) * self.lattice.radius / np.sqrt(3)
            pygame.draw.circle(surface, (255, 0, 0), (int(x), int(y)), 10)
        for (i, j) in self.sources:
            x = (2*j+1) * self.lattice.radius if i % 2 == 0 else \
                (2 * j+2) * self.lattice.radius
            y = (2+3*i) * self.lattice.radius / np.sqrt(3)
            pygame.draw.circle(surface, (0, 255, 0), (int(x), int(y)), 10)
