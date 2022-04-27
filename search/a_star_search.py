"""
@author: Diego Fernández Sebastian  100387203
@author: Ricardo Grande Cros  100386336
@University: uc3m
"""

from copy import deepcopy
from game import Actions, Configuration, Grid, Directions
from util import PriorityQueueWithFunction, manhattanDistance
from search.utils.floyd_warshall import FloydWarshall
import numpy as np
from timeit import default_timer as timer


class PacmanNode(object):
    __slots__ = ("_parent", "_pacman_configuration", "_g", "_h", "_ghosts", "_f")
    # representamos el mapa actual como variable de clase común para todos los nodos.
    GRID = []

    @classmethod
    def _set_grid(cls, grid):
        PacmanNode.GRID = Grid.deepCopy(grid)

    def _set_ghost_pos(self, current_ghosts):
        self._ghosts = deepcopy(current_ghosts[0])
        for i in range(0, len(current_ghosts[1])):
            if current_ghosts[1][i] is False:
                self._ghosts[i] = None

    def __init__(self, prev_state=None, pacman=None, ghosts=None):
        if prev_state is not None:
            self._parent = prev_state
            self._pacman_configuration = Configuration(prev_state.pacman_configuration.pos,
                                                       prev_state.pacman_configuration.direction)
            self._g = prev_state.g
            self._ghosts = deepcopy(prev_state.ghosts)
            self._h = 0
        else:
            self._parent = None
            self._pacman_configuration = Configuration(pacman.pos, pacman.direction)
            self._g = 0
            self._h = 0
            self._set_ghost_pos(ghosts)
        self._f = 0

    @property
    def ghosts(self):
        return self._ghosts

    @property
    def parent(self):
        return self._parent

    @property
    def pacman_configuration(self):
        return self._pacman_configuration

    @pacman_configuration.setter
    def pacman_configuration(self, new_direction):
        """
        Given a new direction, calculate the new position as a tuple and update
        the pacman configuration values with these new values.
        Furthermore, If the new position reached is equal any ghost position, update that
        the ghost with the same position has been slain by pacman.
        """
        new_pos = None
        if new_direction == Directions.NORTH:
            new_pos = (self.pacman_configuration.pos[0], self.pacman_configuration.pos[1] + 1)
        elif new_direction == Directions.SOUTH:
            new_pos = (self.pacman_configuration.pos[0], self.pacman_configuration.pos[1] - 1)
        elif new_direction == Directions.EAST:
            new_pos = (self.pacman_configuration.pos[0] + 1, self.pacman_configuration.pos[1])
        elif new_direction == Directions.WEST:
            new_pos = (self.pacman_configuration.pos[0] - 1, self.pacman_configuration.pos[1])
        elif new_direction == Directions.STOP:
            new_pos = (self.pacman_configuration.pos[0], self.pacman_configuration.pos[1])
        self._pacman_configuration.pos = new_pos
        self._pacman_configuration.direction = new_direction
        # in order not to slain more than 1.
        for i in range(0, len(self.ghosts)):
            if self.ghosts[i] is not None:
                if self.ghosts[i][0] == self._pacman_configuration.pos[0] and self.ghosts[i][1] == \
                        self._pacman_configuration.pos[1]:
                    self.ghosts[i] = None
                    break

    @property
    def f(self):
        return self._f

    def update_f(self):
        #print(self._h, self.g)
        self._f = self.h + self.g

    @property
    def g(self):
        return self._g

    @property 
    def h(self):
        return self._h

    @h.setter
    def h(self, new_heuristic_value):
        self._h = new_heuristic_value

    def update_g(self):
        """ Each step has always cost 1. """
        self._g += 1

    def __eq__(self, other):
        """
        Check whether two states are equal or not. In order to be equal,
        the two states must have the same pacman position and the same
        ghosts slain.
        """
        equal = False
        if isinstance(other, PacmanNode):
            if other.pacman_configuration.pos[0] == self.pacman_configuration.pos[0] and other.pacman_configuration.pos[1] == self.pacman_configuration.pos[1] and self.pacman_configuration.direction == other.pacman_configuration.direction:
                equal = True
                looped = True
                for i in range(0, len(self.ghosts)):
                    if self.ghosts[i] is None and other.ghosts[i] is not None or self.ghosts[i] is not None and other.ghosts[i] is None:
                        looped = False
                        break
                    elif self.ghosts[i] is not None and other.ghosts[i] is not None:
                        if self.ghosts[i][0] != other.ghosts[i][0] or self.ghosts[i][1] != other.ghosts[i][1]:
                            looped = False
                            break
                equal = looped
        return equal

    def __hash__(self):
        data = str(self.pacman_configuration.pos) + str(self.pacman_configuration.direction) + str(self.ghosts)
        return hash(data)

    def __str__(self):
        return f"Node has the following pacman {self.pacman_configuration.pos} & " \
               f"{self.pacman_configuration.direction} with a f cost of {self.f}. Ghosts slain {self.ghosts}. " \
               f" h cost: {self.h} g cost {self.g}."

    def deep_copy_pacman_node(self):
        return PacmanNode(self)


class EnhancedPacmanNode(PacmanNode):

    """
    A specific class used to represent A* search using the 
    tuned manhattan distance of the closest ghost with the relaxed 
    manhattan distance as heuristic function. 
    """
    __slots__ = '_path'

    def __init__(self, prev_state=None, pacman=None, ghosts=None, path=None):
        super().__init__(prev_state, pacman, ghosts)
        self._path = deepcopy(path)
        #print(f"node with path {self._path}")
    
    @property
    def pacman_configuration(self):
        return super().pacman_configuration

    @pacman_configuration.setter
    def pacman_configuration(self, new_direction):
        PacmanNode.pacman_configuration.fset(self, new_direction)
        # delete from the path if the new pos belongs to the new path.
        index = -1
        for i in range(0, len(self._path)):
            if self._path[i][0] == self.pacman_configuration.pos[0] and self._path[i][1] == self.pacman_configuration.pos[1]:
                index = i
        if index != -1:
            self._path.pop(index)
            self._path = self._path[:index]


    def deep_copy_pacman_node(self):
        return EnhancedPacmanNode(self, path=self._path)


class EnhancedPacmanNode2(PacmanNode):
    """
    A specific class used to represent A* search using the cache paths to all ghosts as heuristic function.
    """
    __slots__ = '_paths'

    def __init__(self, prev_state=None, pacman=None, ghosts=None, paths=None):
        super().__init__(prev_state, pacman, ghosts)
        self._paths = deepcopy(paths)

    @property
    def pacman_configuration(self):
        return super().pacman_configuration

    @pacman_configuration.setter
    def pacman_configuration(self, new_direction):
        """
        Due to the fact that the A* search is performed in every tick with ghosts frozen, 
        we only need to update pacman position according to the paths.
        """
        PacmanNode.pacman_configuration.fset(self, new_direction)
        # delete from the path if the new pos belongs to the new path.
        indices = [-1 for _ in range(0, len(self._paths))]
        for i in range(0, len(self._paths)):
            for j in range(0, len(self._paths[i])):
                if self._paths[i][j][0] == self.pacman_configuration.pos[0] and self._paths[i][j][1] == self.pacman_configuration.pos[1]:
                    indices[i] = j
        for i in range(0, len(indices)):
            if indices[i] == -1:
                # The position is not in the path, add to the path to increase its value.
                self._paths[i].append(self.pacman_configuration.pos)
            elif indices[i] == len(self._paths[i])-2:
                # Delete the 2 last position because its in the path.
                # first is the current pos. Latter the new pos reached.
                self._paths[i].pop(-1)
                self._paths[i].pop(-1)
            else:
                # La coincidencia ocurre en mitad del path calculado, eliminar todas las
                # posiciones para actualizar el path.
                self._paths[i] = self._paths[i][:indices[i]]

    def deep_copy_pacman_node(self):
        return EnhancedPacmanNode2(self, paths=self._paths)

    def __str__(self):
        return super().__str__() + f"{self._paths}"


"""
Heuristic Cost Functions for Pacman.
"""


def heuristic_function(state, heuristic: str):
    if heuristic == "floyd_warshall":
        return heuristic_function_floyd_warshall(state)
    elif heuristic == "enhanced_floyd_warshall":
        return enhanced_heuristic_function_floyd_warshall(state)
    elif heuristic == "min_path":
        return heuristic_min_path(state)


def heuristic_min_path(state):
    #print(f"funcion heuristica calculo H {min(len(p) for p in state._paths)}")
    return min(len(p) for p in state._paths)


def cost_function_min_path(state):
    #print(type(state))
    #print(state._paths)
    #print("min path: cost function for root", state.g + min(len(p) for p in state._paths))
    return state.g + min(len(p) for p in state._paths)


def heuristic_function_floyd_warshall(state: PacmanNode):
    distance_to_closest_ghost = np.inf
    index_to_closest_ghost = -1
    count_ghosts = 0
    for i in range(0, len(state.ghosts)):
        if state.ghosts[i] is not None:
            count_ghosts += 1
            current_distance = manhattanDistance(state.pacman_configuration.pos, state.ghosts[i])
            if current_distance < distance_to_closest_ghost:
                distance_to_closest_ghost = current_distance
                index_to_closest_ghost = i
    if distance_to_closest_ghost == np.inf:
        return 0
    else:
        return distance_to_closest_ghost
    """
    if distance_to_closest_ghost != np.inf and count_ghosts == 1:
        return distance_to_closest_ghost
    else:
        ordered_ghosts = [state.ghosts[index_to_closest_ghost]]
        for i in range(0, len(state.ghosts)):
            if i != index_to_closest_ghost:
                ordered_ghosts.append(state.ghosts[i])
        heuristic = FloydWarshall(ordered_ghosts)
        heuristic.solve_all_pairs_shorthest_path()
        return distance_to_closest_ghost + heuristic.get_minimum_cost_path_to_all_vertices()
    """

def cost_function_floyd_warshall(state: PacmanNode):
    """
    A function cost with an admissible heuristic.
    This cost function works as:
    0-. Gets the G cost from the state.
    1-. Try to figure out the distance to the closest ghost.
        1.1-. If the distance to closest ghost != np.inf and num ghosts >= 2: then go 2-.
        1.2-. If the distance to closest ghost != np.inf and num_ghosts == 1: return distance to closest_ghost as H.
        1.3-. If the distance to closest ghost == np.inf, num ghosts must be 0, thus return 0 as H.
    2-. Represent all ghosts in a graph and apply floyd warshall to get all pair shortest path
        in order to get the path of minimum cost.
        The closest ghost will be the start vertex from which Floyd Warshall will be applied in order to get the
        minimum cost path from it to the rest of ghosts.

    """
    total_cost = state.g
    distance_to_closest_ghost = np.inf
    index_to_closest_ghost = -1
    count_ghosts = 0
    for i in range(0, len(state.ghosts)):
        if state.ghosts[i] is not None:
            count_ghosts += 1
            current_distance = manhattanDistance(state.pacman_configuration.pos, state.ghosts[i])
            if current_distance < distance_to_closest_ghost:
                distance_to_closest_ghost = current_distance
                index_to_closest_ghost = i
    if distance_to_closest_ghost == np.inf:
        return total_cost
    else:
        return total_cost + distance_to_closest_ghost
    """if distance_to_closest_ghost != np.inf and count_ghosts == 1:
        return total_cost + distance_to_closest_ghost
    else:
        ordered_ghosts = [state.ghosts[index_to_closest_ghost]]
        for i in range(0, len(state.ghosts)):
            if i != index_to_closest_ghost:
                ordered_ghosts.append(state.ghosts[i])
        heuristic = FloydWarshall(ordered_ghosts)
        heuristic.solve_all_pairs_shorthest_path()
        return total_cost + distance_to_closest_ghost + heuristic.get_minimum_cost_path_to_all_vertices()
    """


def enhanced_cost_function_floyd_warshall(state: EnhancedPacmanNode):
    total_cost = state.g
    distance_to_closest_ghost = np.inf
    index_to_closest_ghost = -1
    count_ghosts = 0
    for i in range(0, len(state.ghosts)):
        if state.ghosts[i] is not None:
            count_ghosts += 1
            current_distance = manhattanDistance(state.pacman_configuration.pos, state.ghosts[i])
            if current_distance < distance_to_closest_ghost:
                distance_to_closest_ghost = current_distance
                index_to_closest_ghost = i
    if index_to_closest_ghost == np.inf:
        return total_cost
    return total_cost + len(state._path)
    """if distance_to_closest_ghost != np.inf and count_ghosts == 1:
        return total_cost + len(state._path)
    else:
        ordered_ghosts = [state.ghosts[index_to_closest_ghost]]
        for i in range(0, len(state.ghosts)):
            if i != index_to_closest_ghost:
                ordered_ghosts.append(state.ghosts[i])
        heuristic = FloydWarshall(ordered_ghosts)
        heuristic.solve_all_pairs_shorthest_path()
        return total_cost + len(state._path) + heuristic.get_minimum_cost_path_to_all_vertices()"""


def enhanced_heuristic_function_floyd_warshall(state: EnhancedPacmanNode):
    distance_to_closest_ghost = np.inf
    index_to_closest_ghost = -1
    count_ghosts = 0
    for i in range(0, len(state.ghosts)):
        if state.ghosts[i] is not None:
            count_ghosts += 1
            current_distance = manhattanDistance(state.pacman_configuration.pos, state.ghosts[i])
            if current_distance < distance_to_closest_ghost:
                distance_to_closest_ghost = current_distance
                index_to_closest_ghost = i
    if distance_to_closest_ghost == np.inf:
        return 0
    else:
        return len(state._path)
    """if distance_to_closest_ghost != np.inf and count_ghosts == 1:
        return len(state._path)
    else:
        ordered_ghosts = [state.ghosts[index_to_closest_ghost]]
        for i in range(0, len(state.ghosts)):
            if i != index_to_closest_ghost:
                ordered_ghosts.append(state.ghosts[i])
        heuristic = FloydWarshall(ordered_ghosts)
        heuristic.solve_all_pairs_shorthest_path()
        return len(state._path) + heuristic.get_minimum_cost_path_to_all_vertices()"""

"""
    A* Algorithm
"""


class AStarAlgorithm(object):            

    def __init__(self, 
                    root,
                    cost_function_used, 
                    hname=None,
                    stats=0):
        self._open_list = PriorityQueueWithFunction(cost_function_used)
        self._closed_list = set()
        self._cost_function_used = cost_function_used
        self._hname = hname
        root.h = self._cost_function_used(root)
        root.update_f()
        self._open_list.push(root)
        # In order to reduce the search space. Once a ghost has been slain, The algorithm stops searching.
        self._kill_ghosts_goal = min(sum(g is None for g in root.ghosts) + 1, len(root.ghosts))
        # stats to get some insight from A*
        self._stats = stats

    def solve(self):
        if self._stats == 1:
            statistics = {"nodes_expanded": 0, "execution_time": 0}
            start = timer()
        solution_node = None
        while not self._open_list.isEmpty():
            current = self._open_list.pop()
            #print(current)
            if sum(g is None for g in current.ghosts) == self._kill_ghosts_goal:
                # its a goal node.
                solution_node = current
                break
            else:
                expand = True
                if current in self._closed_list:
                    #print("repetido!!!!!!!!!!")
                    iterator = iter(self._closed_list)
                    item = next(iterator, None)
                    while current != item:
                        item = next(iterator, None)
                    if current.f < item.f:
                        self._closed_list.remove(item)
                        self._closed_list.add(current)
                    else:
                        # do not expand the node.
                        expand = False
                else:
                    self._closed_list.add(current)
                # get all children.
                if expand:
                    if self._stats == 1: statistics['nodes_expanded'] += 1
                    for dir in Actions.getPossibleActions(current.pacman_configuration, PacmanNode.GRID):
                        if dir != Directions.STOP:
                            child = current.deep_copy_pacman_node()
                            child.pacman_configuration = dir
                            # after changing the pos, evaluate.
                            child.update_g()
                            child.h = heuristic_function(child, self._hname)
                            child.update_f()
                            # print(f"child cst function {self._cost_function(child)}")
                            # print(f"generating successor... @ dir {dir} with f {child.f} and g: {child.g}")
                            self._open_list.push(child)
        # get the path followed.
        prev_action = None
        cost_node = 0
        while solution_node.parent is not None:
            prev_action = solution_node.pacman_configuration.direction
            cost_node = solution_node.f
            solution_node = solution_node.parent

        if self._stats:
            end = timer()
            # time in seconds.
            statistics['execution_time'] = end - start
            print(f"StatsForSolve;Time(s),{statistics['execution_time']};ExpandedNodes,{statistics['nodes_expanded']}")
        return prev_action

    def enhanced_solve(self):
        if self._stats == 1:
            statistics = {"nodes_expanded": 0, "execution_time": 0}
            start = timer()
        solution_node = None
        while not self._open_list.isEmpty():
            current = self._open_list.pop()
            #print(current)
            if sum(g is None for g in current.ghosts) == self._kill_ghosts_goal:
                solution_node = current
                break
            else:
                expand = True
                if current in self._closed_list:
                    iterator = iter(self._closed_list)
                    item = next(iterator, None)
                    while current != item:
                        item = next(iterator, None)
                    if current.f < item.f:
                        self._closed_list.remove(item)
                        self._closed_list.add(current)
                    else:
                        expand = False
                else:
                    self._closed_list.add(current)
                # get all children.
                if expand:
                    if self._stats == 1: statistics['nodes_expanded'] += 1
                    for dir in Actions.getPossibleActions(current.pacman_configuration, PacmanNode.GRID):
                        if dir != Directions.STOP:
                            child = current.deep_copy_pacman_node()
                            child.pacman_configuration = dir
                            # after changing the pos, evaluate.
                            child.update_g()
                            child.h = heuristic_function(child, self._hname)
                            #print(f"Valor calculado {child.h}")
                            #print(f"Valor calculado directo {heuristic_function(child, self._hname)}")

                            child.update_f()
                            self._open_list.push(child)
        # get the path followed.
        prev_action = None
        cost_node = 0
        while solution_node.parent is not None:
            prev_action = solution_node.pacman_configuration.direction
            cost_node = solution_node.f
            solution_node = solution_node.parent

        if self._stats:
            end = timer()
            # time in seconds.
            statistics['execution_time'] = end - start
            print(f"StatsForEnhancedSolve;Time(s),{statistics['execution_time']};ExpandedNodes,{statistics['nodes_expanded']}")

        # print(f"the action to perform is: {prev_action} and cost {cost_node}")
        return prev_action

