"""
@author: Diego Fernández Sebastian  100387203
@author: Ricardo Grande Cros  100386336
@University: uc3m

    It's deprecated. It's no longer been used due to the fact
    that the goal state is represented as a single ghost catch.
    Therefore, there is no need to figure out the shorthest paths
    among other ghosts. 
"""

import numpy as np


class Edge(object):
    """
    Does not support negative weights. 
    """
    __slots__ = ("__from", "__to", "__weight")
    def __init__(self, from_vi: np.uint64, to_vj:np.uint64, weight: np.uint64):
        self.__from = from_vi
        self.__to = to_vj
        self.__weight = weight

    @property
    def from_v(self):
        return self.__from

    @property
    def to_v(self):
        return self.__to
    
    @property
    def weight(self):
        return self.__weight

    def __str__(self):
        return f"the edge goes from {self.__from} to {self.__to} with a cost of {self.__weight}."

    def __eq__(self, other):
        if isinstance(other, Edge):
            if self.__from == other.__from and self.__to == other.__to and self.__weight == other.__weight:
                return True
        return False

class DirectedEdge(Edge):
    """
        ith: from a vertex could be multiple edges going from it to another vertex.
        Thus, we do have a ref count to indicate this case.
    """
    __slots__ = ("__ith")
    def __init__(self, from_vi, to_vj, weight, ith):
        super().__init__(from_vi, to_vj, weight)
        self.__ith = ith

    def __str__(self):
        return super().__str__() + f" and it's the {self.__ith} edge."

class UndirectedEdge(Edge):
    __slots__ = ()
    def __init__(self, from_vi, to_vj, weight):
        super().__init__(from_vi, to_vj, weight)

    def __str__(self):
        return super().__str__()

    def __eq__(self, other):
        return super().__eq__(other)
