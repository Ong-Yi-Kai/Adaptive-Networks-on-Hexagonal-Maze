import numpy as np
import pygame
from typing import Tuple, List


class Lattice:
    """
    Hexagonal lattice using offset coordinates, shoving odd rows to the right
    (Start count with 1).

    For each hexagon tile, we label edges 0-5 starting from the right clockwise
    """

    radius: float = 15  # outer radius of hexagon

    def __init__(self, N: int, M: int):
        """
        N: Number of rows
        M: Number of columns
        """
        self.N = N
        self.M = M

        self.verts = np.ones((N, M))

        # N x M x 6 array of neighbours (3rd dim for 6 directions)
        self.edge_weight = None
        self.edges = None
        self.generate_maze()

    def generate_maze(self):
        """
        Use backtrack algorithm to generate maze. 
        Edges will all be zeros, everytime we visit a node, we set the edge to 1
        """
        edges = np.zeros((self.N, self.M, 6))
        visited = np.zeros((self.N, self.M), dtype=bool)
        stack = [(0, 0)]
        visited[0, 0] = True
        while len(stack) > 0:
            i, j = stack[-1]
            neighbors = self.get_all_neighbors(i, j)
            unvisited_neighbors = [
                n for n in neighbors if not visited[n[0], n[1]]]

            if len(unvisited_neighbors) > 0:
                ni, nj, k = unvisited_neighbors[np.random.randint(
                    len(unvisited_neighbors))]
                edges[i, j, k] = 1
                edges[ni, nj, (k + 3) % 6] = 1
                visited[ni, nj] = True
                stack.append((ni, nj))
            else:
                stack.pop()
        self.edge_weight = edges.copy()
        self.edges = edges.copy()

    def update_vert_values(self, new_values: np.ndarray):
        """
        Update the vertex values with new values.
        """
        if new_values.shape != (self.N, self.M):
            raise ValueError(
                f"New values must have the same shape as the lattice. Need {self.N, self.M}, got {new_values.shape}.")
        self.verts = new_values.copy()

    def update_edge_values(self, new_values: np.ndarray):
        """
        Update the edge values with new values.
        """
        if new_values.shape != (self.N, self.M, 6):
            raise ValueError(
                f"New values must have the same shape as the edges. Need {self.N, self.M, 6}, got {new_values.shape}.")
        self.edge_weight = new_values.copy()

    def get_neighbor(self, i: int, j: int, k: int) -> Tuple[int, int]:
        """
        Get the coordinates of the neighbor in direction k of the hexagon at (i, j).
        """
        new_i, new_j = None, None
        if k == 0:  # right
            new_i, new_j = i, j+1
        elif k == 1:  # bottom-right
            new_i, new_j = (i+1, j) if i % 2 == 0 else (i+1, j+1)
        elif k == 2:  # bottom-left
            new_i, new_j = (i+1, j-1) if i % 2 == 0 else (i+1, j)
        elif k == 3:  # left
            new_i, new_j = i, j-1
        elif k == 4:  # top-left
            new_i, new_j = (i-1, j-1) if i % 2 == 0 else (i-1, j)
        elif k == 5:  # top-right
            new_i, new_j = (i-1, j) if i % 2 == 0 else (i-1, j+1)
        else:
            return None, None
        if new_i < 0 or new_i >= self.N or new_j < 0 or new_j >= self.M:
            return None, None
        return new_i, new_j

    def get_all_neighbors(self, i: int, j: int) -> List[Tuple]:
        """
        Get all the neighbors of the hexagon at (i, j).
        """
        neighbors = []
        for k in range(6):
            ni, nj = self.get_neighbor(i, j, k)
            if ni is not None and nj is not None:
                neighbors.append((ni, nj, k))
        return neighbors

    def draw(self, screen: pygame.Surface):
        for i in range(self.N):
            for j in range(self.M):

                x = (2*j+1) * self.radius if i % 2 == 0 else (2*j+2) * self.radius
                y = (2+3*i) * self.radius / np.sqrt(3)

                v0 = (x + self.radius * np.sqrt(3) / 2, y - self.radius / 2)
                v1 = (x + self.radius * np.sqrt(3) / 2, y + self.radius / 2)
                v2 = (x, y + self.radius)
                v3 = (x - self.radius * np.sqrt(3) / 2, y + self.radius / 2)
                v4 = (x - self.radius * np.sqrt(3) / 2, y - self.radius / 2)
                v5 = (x, y - self.radius)
                vertices = [v0, v1, v2, v3, v4, v5]

                # draw in the hexagon sides that do not have a maze connection
                for k in range(6):
                    if self.edges[i, j, k] == 0:
                        pygame.draw.line(screen, (255, 255, 255),
                                         vertices[k], vertices[(k + 1) % 6], 2)

                # draw edges representing neighbours
                for k in range(6):
                    if self.edge_weight[i, j, k] > 0:
                        x2 = (x + self.radius * np.cos(k * np.pi / 3))
                        y2 = (y + self.radius * np.sin(k * np.pi / 3))

                        c = max(np.tanh(self.edge_weight[i, j, k]) * 255, 0)
                        pygame.draw.line(screen, (0, 0, int(c)),
                                         (x, y), (x2, y2), int(np.tanh(self.edge_weight[i, j, k])*7))

                # draw circle representing pressure at the node
                # v_ij = self.verts[i, j]
                # c = max(np.tanh(v_ij) * 255, 0)
                # pygame.draw.circle(screen, (100, 0, int(c)), (int(x), int(y)),
                #                    5, 0)
