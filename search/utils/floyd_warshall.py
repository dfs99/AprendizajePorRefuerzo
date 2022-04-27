"""
@author: Diego Fernández Sebastian  100387203
@author: Ricardo Grande Cros  100386336
@University: uc3m

    It's deprecated. It's no longer been used due to the fact
    that the goal state is represented as a single ghost catch.
    Therefore, there is no need to figure out the shorthest paths
    among other ghosts. 

"""

from search.utils.edges import UndirectedEdge
import numpy as np


class FloydWarshall(object):
    def _get_matrix_vertices_from_ghosts(self, ghosts: list):
        num_ghosts = 0
        for ghost in ghosts:
            if ghost is not None:
                num_ghosts += 1
        return np.full((num_ghosts, num_ghosts), np.inf), num_ghosts

    def _get_undirected_edges_from_ghosts(self, ghosts: list):
        edges = []
        ghosts_cleaned = [x for x in ghosts if x is not None]
        for i in range(0, len(ghosts_cleaned)-1):
            edges.append(UndirectedEdge(i, i+1,
             abs(ghosts_cleaned[i][0] - ghosts_cleaned[i+1][0]) + 
             abs(ghosts_cleaned[i][1] - ghosts_cleaned[i+1][1]))) # using manhattan distance
        return edges

    def __init__(self, ghosts: list):
        self.__distances, self.__num_vertices = self._get_matrix_vertices_from_ghosts(ghosts)
        self.__edges = self._get_undirected_edges_from_ghosts(ghosts)

    def solve_all_pairs_shorthest_path(self):
        for i in range(0, self.__num_vertices):
            self.__distances[i][i] = 0

        for edge in self.__edges:
            self.__distances[edge.from_v][edge.to_v] = edge.weight
            self.__distances[edge.to_v][edge.from_v] = edge.weight
        
        for k in range(0, self.__num_vertices):
            for i in range(0, self.__num_vertices):
                for j in range(0, self.__num_vertices):
                    self.__distances[i][j] = min(self.__distances[i][j], 
                        self.__distances[i][k] + self.__distances[k][j])

    def get_minimum_cost_path_to_all_vertices(self):
        """It will be the sum of the diagonal up to the main diagonal."""
        minimum_distance = 0
        for i in range(0, self.__num_vertices-1):
            minimum_distance += self.__distances[i+1][i]
        return minimum_distance

    def __str__(self):
        msg = "Matrix that represent all-pairs shorthest paths:\n"
        for vector in self.__distances:
            for value in vector:
                msg += ( str(value) + " ")
            msg += "\n"
        msg += "\n"
        return msg

