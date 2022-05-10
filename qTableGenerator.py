import math

# ATTRIBUTES & CARDINALITY
quadrantGhosts = 8
quadrantFood = 9
discreteDistance = 3
wallType = 13


# num actions.
NUM_ACTIONS = 5
attributes = [quadrantGhosts, quadrantFood, wallType]
TABLE_ROWS = math.prod(attributes)


# num rows.
def generate_qtable(num_actions, num_attributes, path):
    with open(path, 'w', encoding='utf-8') as qTable:
        for row in range(num_attributes):
            qTable.write('0.0 ' * num_actions + '\n')


generate_qtable(NUM_ACTIONS, TABLE_ROWS, './AprendizajePorRefuerzo/qtable.txt')
