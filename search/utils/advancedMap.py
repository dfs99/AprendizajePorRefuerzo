"""
@author: Diego Fernández Sebastian  100387203
@author: Ricardo Grande Cros  100386336
@University: uc3m
"""

from copy import deepcopy
import re
from numpy import inf
from timeit import default_timer as timer


class Steps(object):
    """
    The class Steps represents the attributes needed to shape a node.
    """
    __slots__ = ('_g', '_pos', '_dir', '_parent')

    def _update_pos(self, move, other):
        new_pos = None
        if move == 'LEFT':
            new_pos = (other._pos[0] - 1, other._pos[1])
        elif move == 'RIGHT':
            new_pos = (other._pos[0] + 1, other._pos[1])
        elif move == 'UP':
            new_pos = (other._pos[0], other._pos[1] + 1)
        elif move == 'DOWN':
            new_pos = (other._pos[0], other._pos[1] - 1)
        return new_pos
    
    def __init__(self, pos=None, move=None, other=None):
        if pos is not None:
            self._g = 0
            self._pos = pos
            self._dir = move
            self._parent = None
        else:
            self._g = other._g + 1
            self._pos = self._update_pos(move, other)
            self._parent = other
            self._dir = move

    def __eq__(self, other):
        if isinstance(other, Steps):
            if self._pos[0] == other._pos[0] and self._pos[1] == other._pos[1]:
                return True
        return False 
    
    def __str__(self):
        return f"Step node has g cost {self._g} with a position {self._pos}"

    def __hash__(self):
        data = str(self._pos[0]) + str(self._pos[1])
        return hash(data)


class AdvancedMap(object):
    """
        The class is used to map the entire map in order to cache 
        the maze and ghosts.
    """
    def _convert_boolean_map(self, boolean_map):
        # fetch map from Game.Grid object.
        out = [[str(boolean_map.data[x][y])[0] for x in range(boolean_map.width)] for y in range(boolean_map.height)]
        out.reverse()
        map = ([''.join(x) for x in out])
        # Compress the map deleting redundant walls and ghost dead places.
        pattern = 'T{'+str(boolean_map.width)+'}'
        max_walls = 2
        compresed_map = []
        for row in map:
            if max_walls == 0: break
            compresed_map.append(row)
            if re.match(pattern, row) and max_walls != 0:
                max_walls -= 1
        # Get an advanced map painting its coordinates and walls.
        start_j = 0
        start_i = boolean_map.height - 2
        advanced_map = []
        for i in range(1, len(compresed_map)-1):
            new_row_advanced = []
            for j in range(0, len(compresed_map[0])):
                if compresed_map[i][j] != 'T':
                    new_row_advanced.append((start_j, start_i))
                else:
                    if j != 0 and j != len(compresed_map[0])-1:
                        new_row_advanced.append('T')
                start_j += 1
            advanced_map.append(new_row_advanced)
            start_j = 0
            start_i -= 1
        return advanced_map 

    def __init__(self, boolean_map):
        self._map = self._convert_boolean_map(boolean_map)

    def cache_lookup(self, pacman_pos, ghosts_pos):
        """
        Tuned Branch & Bound algorithm to find best paths to all ghosts at first. 
        """
        ghosts_indices = [self._get_index_from_pos(ghost_pos) for ghost_pos in ghosts_pos]
        root = Steps(pos=pacman_pos)
        stack = [root]
        best_values = [inf for _ in ghosts_pos]
        best_states = [None for _ in ghosts_pos]
        already_visited = set()
        # Get all paths
        while len(stack) > 0:
            current = stack.pop(0)
            already_visited.add(current)
            current_indices = self._get_index_from_pos(current._pos)
            for i in range(0, len(ghosts_indices)):
                if current_indices[0] == ghosts_indices[i][0] and current_indices[1] == ghosts_indices[i][1]:
                    if best_values[i] > current._g:
                        best_states[i] = current
                        best_values[i] = current._g
            for move in self._get_moves_from_pos(current_indices):
                child = Steps(move=move, other=current)
                if child not in already_visited:
                    stack.insert(0, child)
                else:
                    for s in already_visited:
                        if s == child and s._g > child._g:
                            stack.insert(0, child)
                            already_visited.remove(s)
                            already_visited.add(child)
        # get paths.
        paths = []
        for i in range(0, len(best_states)):
            paths.append([])
            current_state = best_states[i]
            while current_state is not None:
                paths[i].append(current_state._pos)
                current_state = current_state._parent
        return paths

    def refined_manhattan_distance(self, pacman_pos, ghost_pos, stats=0):
        """
        Gets a tuned manhattan distance for a single ghost using Branch & Bound algorithm.
        """
        ghost_indices = self._get_index_from_pos(ghost_pos)
        root = Steps(pos=pacman_pos)
        stack = [root]
        best_value = inf
        best_state = None
        already_visited = set()
        if stats == 1:
            start = timer()
            expanded_nodes = 0
        while(len(stack) > 0):
            current = stack.pop(0)
            already_visited.add(current)
            current_indices = self._get_index_from_pos(current._pos)
            if stats == 1: expanded_nodes += 1
            if (current_indices[0] == ghost_indices[0] and current_indices[1] == ghost_indices[1]):
                if best_value > current._g:
                    best_state = current
                    best_value = current._g
            for move in self._get_moves_from_pos(current_indices):
                child = Steps(move=move, other=current)
                if child._g < best_value:
                    if child not in already_visited:
                        stack.insert(0, child)
                    else:
                        for s in already_visited:
                            if s == child and s._g > child._g:
                                stack.insert(0, child)
                                already_visited.remove(s)
                                already_visited.add(child)
        path = []
        while best_state is not None:
            path.append(best_state._pos)
            best_state = best_state._parent
        if stats == 1:
            end = timer()
            elapsed_time = end - start
            print(f"StatsForOptimizedHeuristicPath;Time(s),{elapsed_time};ExpandedNodes,{expanded_nodes}")
        return path
        
        
    def _get_moves_from_pos(self, pos):
        posible_moves = {'UP', 'DOWN', 'RIGHT', 'LEFT'}
        if pos[0] == 0:
            posible_moves.remove('UP')
        if pos[1] == 0:
            posible_moves.remove('LEFT')
        if pos[0] == len(self._map) - 1:
            posible_moves.remove('DOWN')
        if pos[1] == len(self._map[0]) - 1:
            posible_moves.remove('RIGHT')
        # Check inner walls using the remaining posible moves.
        moves = deepcopy(posible_moves)
        for available_move in posible_moves:
            if available_move == 'UP' and self._map[pos[0] - 1][pos[1]] == 'T':
                moves.remove('UP')
            elif available_move == 'DOWN' and self._map[pos[0] + 1][pos[1]] == 'T':
                moves.remove('DOWN')
            elif available_move == 'RIGHT' and self._map[pos[0]][pos[1] + 1] == 'T':
                moves.remove('RIGHT')
            elif available_move == 'LEFT' and self._map[pos[0]][pos[1] - 1] == 'T':
                moves.remove('LEFT')
        return list(moves)

    def _get_index_from_pos(self, pos):
        for i in range(0, len(self._map)):
            for j in range(0, len(self._map[0])):
                if self._map[i][j] != 'T' and self._map[i][j][0] == pos[0] and self._map[i][j][1] == pos[1]:
                    return i,j
        return ValueError("Index not found @ AdvancedMap.py")

