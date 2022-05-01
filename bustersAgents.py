from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
from re import T
from tkinter import MULTIPLE, N
from numpy import inf
from search.utils.advancedMap import AdvancedMap
from search.a_star_search import AStarAlgorithm, PacmanNode, EnhancedPacmanNode, cost_function_floyd_warshall, \
    enhanced_cost_function_floyd_warshall, EnhancedPacmanNode2, cost_function_min_path
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters


class NullGraphics(object):
    "Placeholder for graphics"

    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True,
                 elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        # for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        # self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."
    LAST_SCORE = 0

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

    def printLineData(self, gameState):
        """ Los datos interesantes a almacenar son: """
        """
        # Pacman position X
        result = str(gameState.getPacmanPosition()[0])
        # Pacman position Y
        result += "," +str(gameState.getPacmanPosition()[1])
        # Ghosts alive
        result += "," + str(0 if not gameState.getLivingGhosts()[1] else 1)
        result += "," + str(0 if not gameState.getLivingGhosts()[2] else 1)
        result += "," + str(0 if not gameState.getLivingGhosts()[3] else 1)
        result += "," + str(0 if not gameState.getLivingGhosts()[4] else 1)

        # Ghosts positions
        result += "," + str(gameState.getGhostPositions()[0][0])
        result += "," + str(gameState.getGhostPositions()[0][1])
        result += "," + str(gameState.getGhostPositions()[1][0])
        result += "," + str(gameState.getGhostPositions()[1][1])
        result += "," + str(gameState.getGhostPositions()[2][0])
        result += "," + str(gameState.getGhostPositions()[2][1])
        result += "," + str(gameState.getGhostPositions()[3][0])
        result += "," + str(gameState.getGhostPositions()[3][1])

        # Distance to ghosts taking walls into account
        from copy import deepcopy
        aMap = AdvancedMap(gameState.data.layout.walls)
        living = []
        indices_living = set()
        for i in range(0, len(gameState.getGhostPositions())):
            if gameState.getLivingGhosts()[1::][i] is True:
                living.append(gameState.getGhostPositions()[i])
                indices_living.add(i)
        all_paths = aMap.cache_lookup(gameState.getPacmanPosition(), living)
        paths = deepcopy(gameState.getLivingGhosts()[1::])
        for i in range(len(paths)):
            if paths[i] and len(all_paths) > 0:
                paths[i] = len(all_paths[0])
                all_paths.pop(0)

        result += "," + str(-1 if paths[0] is False else paths[0])
        result += "," + str(-1 if paths[1] is False else paths[1])
        result += "," + str(-1 if paths[2] is False else paths[2])
        result += "," + str(-1 if paths[3] is False else paths[3])


        # Relative distances
        for ghost_position in gameState.getGhostPositions():
            result += "," + str(gameState.getPacmanPosition()[0]-ghost_position[0])
            result += "," + str(gameState.getPacmanPosition()[1]-ghost_position[1])

        # Acciones legales del pacman
        result += "," + str(1 if 'North' in gameState.getLegalPacmanActions() else 0)
        result += "," + str(1 if 'South' in gameState.getLegalPacmanActions() else 0)
        result += "," + str(1 if 'East' in gameState.getLegalPacmanActions() else 0)
        result += "," + str(1 if 'West' in gameState.getLegalPacmanActions() else 0)

        # Score
        result += "," + str(self.LAST_SCORE)
        # Next Score
        result += "," + str(gameState.getScore())
        BasicAgentAA.LAST_SCORE = gameState.getScore()

        # Pacman direction (dependant variable)
        pacman_dir = gameState.data.agentStates[0].getDirection()
        if pacman_dir == 'North':
            result += ",1"
        elif pacman_dir == 'South':
            result += ",2"
        elif pacman_dir == 'East':
            result += ",3"
        elif pacman_dir == 'West':
            result += ",4"
        else:
            result += ",5" # STOP
    
        return result
        """
        return None



from distanceCalculator import Distancer, manhattanDistance
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''


class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        # print(gameState.getLegalActions(0)) # devuelve una lista con todas las opciones válidas.
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:  move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal: move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:   move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i + 1]]
        return Directions.EAST


class BasicAgentAA(BustersAgent):

    """
    Within BasicAgentAA the intelligent approach has been implemented
    mainly using informed search.

    Class Variables:
        -> ADVANCED_MAP: stores an advancedMap instance that contains a detailed 
                         representation of the map with useful functions.
        -> CACHE_MAP: stores all shorthest paths to all lived ghosts. Note that these
                      paths are modified as the game runs. Thus, the path might loose
                      its optimality. If this operation is performed for each turn, 
                      the pacman will perform all optimum movements.
        -> ALREADY_ERASED: stores a short-term memory of the ghosts already erased from
                           the CACHE_MAP due to the fact that they have been slain.
        -> ONCE_TIME_VARS: used to activate once some setting up.
    """
    
    ADVANCED_MAP = None
    CACHE_MAP = None
    ALREADY_ERASED = None
    ONCE_TIME_VARS = False
    LAST_SCORE = 0

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ",
              [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # print("Diego - distances: ", [x if x is not None else 99999 for x in gameState.data.ghostDistances])
        # print("Diego  -  Target ghost: ", np.argmin([x if x is not None else 99999 for x in gameState.data.ghostDistances]))
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):
        """
        Vars used to perform one action or another one.
        -> OPTIMIZED: values {True: activated; False: deactivated}
        -> MULTIPLE_PATHS_HEURISTIC: values {True: activated; False: deactivated}
                                     If activated, it chooses actions according to paths calculated beforehand.   
        -> PRINT_INFO: values {True: activated; False: deactivated} In order to call "printInfo" function each time. 
        -> PRINT_TRACES: values {0: no; 1: yes} It prints out time and space traces for search algorithms.
        """
        OPTIMIZED = False
        MULTIPLE_PATHS_HEURISTIC = False
        RECALCULATE_EACH_DEATH = False
        PRINT_INFO = False
        PRINT_TRACES = 0
        
        # Default settings for the method.
        self.countActions = self.countActions + 1
        move = Directions.STOP
        
        # update things to run multiple games using cache maps.
        # Code to reset the class variables for each game.
        if BasicAgentAA.CACHE_MAP is not None:
            begin1 = True
            begin2 = False
            for x in gameState.getLivingGhosts()[1::]:
                if x is False:
                    begin1 = False
                    break
            if begin1:
                for x in BasicAgentAA.CACHE_MAP:
                    if x is None:
                        begin2 = True
                        break
            if begin1 and begin2:
                #print("reseteo completo!")
                BasicAgentAA.ADVANCED_MAP = None
                BasicAgentAA.CACHE_MAP = None
                BasicAgentAA.ALREADY_ERASED = None
        
        # Once time vars for each game.
        if BasicAgentAA.ONCE_TIME_VARS is False:
            PacmanNode._set_grid(gameState.data.layout.walls)
            BasicAgentAA.ONCE_TIME_VARS = True

        if PRINT_INFO: self.printInfo(gameState)
        
        if BasicAgentAA.ALREADY_ERASED is None and MULTIPLE_PATHS_HEURISTIC:
            BasicAgentAA.ALREADY_ERASED = [False for _ in range(0, len(gameState.getGhostPositions()))]
        
        
        if MULTIPLE_PATHS_HEURISTIC:
            
            if BasicAgentAA.CACHE_MAP is not None:
                # update ghost positions over paths.
                #print(f"Previous Paths:")
                #for p in BasicAgentAA.CACHE_MAP:
                #    print(p)
                for i in range(0, len(gameState.getGhostPositions())):
                    index = -1
                    index_pacman = -1
                    if gameState.getLivingGhosts()[1::][i] is True:
                        # update ghosts positions
                        for j in range(0, len(BasicAgentAA.CACHE_MAP[i])):
                            if BasicAgentAA.CACHE_MAP[i][j][0] == gameState.getGhostPositions()[i][0] and \
                                    BasicAgentAA.CACHE_MAP[i][j][1] == gameState.getGhostPositions()[i][1]:
                                index = j
                        if index == -1:
                            # añadir al path la nueva posición del fantasma.
                            BasicAgentAA.CACHE_MAP[i].insert(0, gameState.getGhostPositions()[i])
                        elif index != 0:
                            # BasicAgentAA.CACHE_MAP[i].pop(0)
                            # used to delete cycles. For instance
                            # [(10, 3), (11, 3), (11, 4), (10, 4), (9, 4), (8, 4), (7, 4), ...
                            # and the current position now is (10,4); Thus update the position deleting the first 3 elements
                            BasicAgentAA.CACHE_MAP[i] = BasicAgentAA.CACHE_MAP[i][index:]

                        # update pacman position
                        for j in range(0, len(BasicAgentAA.CACHE_MAP[i])):
                            if BasicAgentAA.CACHE_MAP[i][j][0] == gameState.getPacmanPosition()[0] and \
                                    BasicAgentAA.CACHE_MAP[i][j][1] == gameState.getPacmanPosition()[1]:
                                index_pacman = j
                                break
                        if index_pacman == -1:
                            BasicAgentAA.CACHE_MAP[i].append(gameState.getPacmanPosition())
                        else:
                        #elif index_pacman == (len(BasicAgentAA.CACHE_MAP[i]) - 2):
                            #BasicAgentAA.CACHE_MAP[i].pop(-1)
                            BasicAgentAA.CACHE_MAP[i] = BasicAgentAA.CACHE_MAP[i][:index_pacman]
                        
                #print(f"updated paths:")
                #for p in BasicAgentAA.CACHE_MAP:
                #    print(p)
            
            if BasicAgentAA.ADVANCED_MAP is None and BasicAgentAA.CACHE_MAP is None:
                BasicAgentAA.ADVANCED_MAP = AdvancedMap(gameState.data.layout.walls)
                living = []
                for i in range(0, len(gameState.getGhostPositions())):
                    if gameState.getLivingGhosts()[1::][i] is True:
                        living.append(gameState.getGhostPositions()[i])
                BasicAgentAA.CACHE_MAP = BasicAgentAA.ADVANCED_MAP.cache_lookup(gameState.getPacmanPosition(), living)

            # se actualizan los fantasmas muertos
            for i in range(0, len(gameState.getLivingGhosts()[1::])):
                if gameState.getLivingGhosts()[1::][i] is False and BasicAgentAA.ALREADY_ERASED[i] is False:
                    BasicAgentAA.CACHE_MAP[i] = None
                    BasicAgentAA.ALREADY_ERASED[i] = True
                    # Si activada la opción, volver a recalcular los paths para que sea óptimo.
                    if RECALCULATE_EACH_DEATH:
                        living = []
                        death_indices = []
                        for i in range(0, len(gameState.getGhostPositions())):
                            if gameState.getLivingGhosts()[1::][i] is True:
                                living.append(gameState.getGhostPositions()[i])
                            else:
                                death_indices.append(i)
                        BasicAgentAA.CACHE_MAP = BasicAgentAA.ADVANCED_MAP.cache_lookup(gameState.getPacmanPosition(), living)
                        for index in death_indices:
                            BasicAgentAA.CACHE_MAP.insert(index, None)
                        


            paths_to_state = [x for x in BasicAgentAA.CACHE_MAP if x is not None]
            root = EnhancedPacmanNode2(pacman=gameState.getPacmanState().configuration,
                                    ghosts=(gameState.getGhostPositions(), gameState.getLivingGhosts()[1::]),
                                    paths=paths_to_state)

            algorithm = AStarAlgorithm(root, 
                            cost_function_min_path,
                            hname='min_path',
                            stats=PRINT_TRACES)
            
            move = algorithm.enhanced_solve()

        # Each time pacman has to choose an action, perform an A*
        # Set the current layout.
        #PacmanNode._set_grid(gameState.data.layout.walls)
        # Generate the root State for A*

        # Uses basic approach.
        if OPTIMIZED is False and MULTIPLE_PATHS_HEURISTIC is False:
            root = PacmanNode(pacman=gameState.getPacmanState().configuration,
                          ghosts=(gameState.getGhostPositions(), gameState.getLivingGhosts()[1::]))
            algorithm = AStarAlgorithm(root, cost_function_floyd_warshall, hname='floyd_warshall', stats=PRINT_TRACES)
            move = algorithm.solve()
        
        if OPTIMIZED:
            # get the closest ghost.
            root = PacmanNode(pacman=gameState.getPacmanState().configuration,
                          ghosts=(gameState.getGhostPositions(), gameState.getLivingGhosts()[1::]))
            closest_ghost = None
            optimized_path = None
            distance_to_closest_ghost = inf
            for i in range(0, len(root.ghosts)):
                if root.ghosts[i] is not None:
                    current_distance = manhattanDistance(root.pacman_configuration.pos, root.ghosts[i])
                    if current_distance < distance_to_closest_ghost:
                        distance_to_closest_ghost = current_distance
                        closest_ghost = root.ghosts[i]
            if closest_ghost is None:
                optimized_path = []
            else:
                if BasicAgentAA.ADVANCED_MAP is None:
                    BasicAgentAA.ADVANCED_MAP = AdvancedMap(gameState.data.layout.walls)
                optimized_path = BasicAgentAA.ADVANCED_MAP.refined_manhattan_distance(root.pacman_configuration.pos,
                                                                                      closest_ghost, stats=PRINT_TRACES)
                # delete the last position because its the current pacman's position
                optimized_path.pop(-1)
            # generate the enhanced node root.
            root = EnhancedPacmanNode(pacman=gameState.getPacmanState().configuration,
                                               ghosts=(gameState.getGhostPositions(), gameState.getLivingGhosts()[1::]),
                                               path=optimized_path)
            algorithm = AStarAlgorithm(root, 
                            enhanced_cost_function_floyd_warshall,
                            hname='enhanced_floyd_warshall',
                            stats=PRINT_TRACES)
            move = algorithm.enhanced_solve()
        
        return move

    def printLineData(self, gameState):
        """ Los datos interesantes a almacenar son: """
        """
        # Pacman position X
        result = str(gameState.getPacmanPosition()[0])
        # Pacman position Y
        result += "," +str(gameState.getPacmanPosition()[1])
        # Ghosts alive
        result += "," + str(0 if not gameState.getLivingGhosts()[1] else 1)
        result += "," + str(0 if not gameState.getLivingGhosts()[2] else 1)
        result += "," + str(0 if not gameState.getLivingGhosts()[3] else 1)
        result += "," + str(0 if not gameState.getLivingGhosts()[4] else 1)

        # Ghosts positions
        result += "," + str(gameState.getGhostPositions()[0][0])
        result += "," + str(gameState.getGhostPositions()[0][1])
        result += "," + str(gameState.getGhostPositions()[1][0])
        result += "," + str(gameState.getGhostPositions()[1][1])
        result += "," + str(gameState.getGhostPositions()[2][0])
        result += "," + str(gameState.getGhostPositions()[2][1])
        result += "," + str(gameState.getGhostPositions()[3][0])
        result += "," + str(gameState.getGhostPositions()[3][1])

        # Distance to ghosts taking walls into account
        from copy import deepcopy
        aMap = AdvancedMap(gameState.data.layout.walls)
        living = []
        indices_living = set()
        for i in range(0, len(gameState.getGhostPositions())):
            if gameState.getLivingGhosts()[1::][i] is True:
                living.append(gameState.getGhostPositions()[i])
                indices_living.add(i)
        all_paths = aMap.cache_lookup(gameState.getPacmanPosition(), living)
        paths = deepcopy(gameState.getLivingGhosts()[1::])
        for i in range(len(paths)):
            if paths[i] and len(all_paths) > 0:
                paths[i] = len(all_paths[0])
                all_paths.pop(0)

        result += "," + str(-1 if paths[0] is False else paths[0])
        result += "," + str(-1 if paths[1] is False else paths[1])
        result += "," + str(-1 if paths[2] is False else paths[2])
        result += "," + str(-1 if paths[3] is False else paths[3])


        # Relative distances
        for ghost_position in gameState.getGhostPositions():
            result += "," + str(gameState.getPacmanPosition()[0]-ghost_position[0])
            result += "," + str(gameState.getPacmanPosition()[1]-ghost_position[1])

        # Acciones legales del pacman
        result += "," + str(1 if 'North' in gameState.getLegalPacmanActions() else 0)
        result += "," + str(1 if 'South' in gameState.getLegalPacmanActions() else 0)
        result += "," + str(1 if 'East' in gameState.getLegalPacmanActions() else 0)
        result += "," + str(1 if 'West' in gameState.getLegalPacmanActions() else 0)

        # Score
        result += "," + str(self.LAST_SCORE)
        # Next Score
        result += "," + str(gameState.getScore())
        BasicAgentAA.LAST_SCORE = gameState.getScore()

        # Pacman direction (dependant variable)
        pacman_dir = gameState.data.agentStates[0].getDirection()
        if pacman_dir == 'North':
            result += ",1"
        elif pacman_dir == 'South':
            result += ",2"
        elif pacman_dir == 'East':
            result += ",3"
        elif pacman_dir == 'West':
            result += ",4"
        else:
            result += ",5" # STOP

        return result"""
        return ""

# todo: clase de agente por aprendizaje por refuerzo.
class QLearningAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.actions = {"North": 0, "East": 1, "South": 2, "West": 3, "Stop": 4}
        self.table_file = open("AprendizajePorRefuerzo/qtable.txt", "r+")
        self.q_table = self.readQtable()
        self.epsilon = 0.3
        self.alpha = 0.5
        self.discount_rate = 0.7
        self.estadoJuego = gameState

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()
        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item) + " ")
            self.table_file.write("\n")

    #         self.table_file_csv.seek(0)
    #         self.table_file_csv.truncate()
    #         for line in self.q_table:
    #             for item in line[:-1]:
    #                 self.table_file_csv.write(str(item)+", ")
    #             self.table_file_csv.write(str(line[-1]))
    #             self.table_file_csv.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")

    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()

    def computePosition(self, state):
        # TODO: MODIFICAR PARA INDEXAR CORRECTAMENTE A LA TABLA.
        """
        Compute the row of the qtable for a given state.
        For instance, the state (3,1) is the row 7
        """
        # return state[0] + state[1] * 4

        # de momento, devolvemos solo el cuadrante del pacman.
        return state

    def getQValue(self, state, action):

        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]

        return self.q_table[position][action_column]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # legalActions = self.getLegalActions(state)
        legalActions = self.estadoJuego.getLegalActions(0)
        if len(legalActions) == 0:
            return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # legalActions = self.getLegalActions(state)
        legalActions = self.estadoJuego.getLegalActions(0)
        if len(legalActions) == 0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        # print("llamando a las acciones desde aqui")
        # Pick Action
        # legalActions = self.getLegalActions(state)
        legalActions = self.estadoJuego.getLegalActions(0)
        action = None

        if len(legalActions) == 0:
            return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            #print("tirando una acción completamente aleatoria...")
            return random.choice(legalActions)
        #return self.getPolicy(state.getQuadrantNearestGhost(self.distancer))
        return random.choice(legalActions)

    def update(self, gamestate, action, nextState, reward):
        # TODO: ESTOS ESTADOS SON GAMESTATE
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        Good Terminal state -> reward 1
        Bad Terminal state -> reward -1
        Otherwise -> reward 0

        Q-Learning update:

        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

        """

        #print(f"Pacman position {gamestate.getPacmanPosition()} -- Nearest Food Quadrant {gamestate.getQuadrantNearestFood(self.distancer)}")
        #print(f"Pacman position {gamestate.getPacmanPosition()} -- Nearest Ghost Quadrant {gamestate.getQuadrantNearestGhost(self.distancer)}")


        # se actualiza el estado del juego en el agente para que pueda tomar las acciones
        # válidas para el siguiente estado.
        self.estadoJuego = nextState
        # TRACE for transition and position to update. Comment the following lines if you do not want to see that trace
        #         print("Update Q-table with transition: ", state, action, nextState, reward)

        # todo: le estamos pasando el cuadrante.
        position = self.computePosition(gamestate.getQuadrantNearestGhost(self.distancer))
        action_column = self.actions[action]
        #
        #         print("Corresponding Q-table cell to update:", position, action_column)

        "*** YOUR CODE HERE ***"
        # todo: la generación de recompensas es irregular. Saca -1, y 4 0's y otro -1 asi todo el rato...
        if reward == 99 or reward == 100:
            reward = 100
        elif reward < 2:
            reward = 0
        elif reward > 250:
            # todo: cuando come un pacman y una comida a la vez.
            reward = 300
        else:
            reward = 200
        #print(f"Comprobacion del reward {reward}")

        # TRACE for updated q-table. Comment the following lines if you do not want to see that trace
        #         print("Q-table:")
        #         self.printQtable()
        if reward > 0:
            # update q(state,action)
            self.q_table[position][action_column] = \
                (1 - self.alpha) * self.q_table[position][action_column] + self.alpha * reward
        else:
            self.q_table[position][action_column] = (1 - self.alpha) * self.q_table[position][action_column] + \
                                                    self.alpha * (reward + self.discount_rate * self.getValue(nextState.getQuadrantNearestGhost(self.distancer)))

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)
