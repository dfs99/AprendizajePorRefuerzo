from matplotlib import pyplot as plt
import csv

class Experimenter:
    def __init__(self, id):
        self.id = id
        self.games = []
        self.scores = []
        self.load_experiment()
        self.old_qtable = []
        self.qtable = []

    def load_experiment(self):
        """ reads experiment csv and loads info in self.games and self.scores """
        try:
            with open(f'./experimenter/{self.id}.csv', 'r') as experiment_info:
                reader = csv.DictReader(experiment_info)
                for row in reader:
                    self.scores.append(int(row['score']))

                print(self.scores)
        except FileNotFoundError:
            self.scores = []



    def plotScores(self):
        x = range(1, len(self.scores)+1)
        y = self.scores
        plt.plot(x,y)
        plt.xticks(x)
        plt.savefig(f'./experimenter/{self.id}')

    def dumpScores(self):
        with open(f'./experimenter/{self.id}.csv', 'w') as experiment_info:
            writer = csv.writer(experiment_info)
            writer.writerow(['game', 'score'])
            writer.writerows(zip([i for i in range(1, len(self.scores)+1)], self.scores))



    def change_qtables(self):
        # read current qtable and store it to old_qtable
        with open('./AprendizajePorRefuerzo/qtable.txt', 'r+') as qtable:
            print("open")
            self.old_qtable = qtable.readlines()
            print(self.old_qtable)

        experiment_qtable = []
        # read experiments qtable, if it doesn't exist create it
        try:
            with open(f'./experimenter/{self.id}.txt', 'r+') as experiment:
                experiment_qtable = experiment.readlines()
        except FileNotFoundError:
            # create qtable
            from qTableGenerator import generate_qtable, NUM_ACTIONS, TABLE_ROWS
            generate_qtable(NUM_ACTIONS, TABLE_ROWS, f'experimenter/{self.id}.txt')
            # call again
            self.change_qtables()

        # write experiments table to qtable
        with open('./AprendizajePorRefuerzo/qtable.txt', 'w+') as qtable:
            qtable.writelines(experiment_qtable)

        print(experiment_qtable)


    def restore_qtable(self):
        print("enter")
        # read q table (experiment's result)
        result = []
        with open('./AprendizajePorRefuerzo/qtable.txt', 'r+') as qtable:
            result = qtable.readlines()


        # TODO: la qtable no se ha actualizado todavía??? wtf, cuándo se actualiza entonces????
        print(result)
        # restore qtable with previous data
        with open('./AprendizajePorRefuerzo/qtable.txt', 'w+') as qtable:
            qtable.writelines(self.old_qtable)

        # write result to experiments file
        with open(f'./experimenter/{self.id}.txt', 'w+') as qtable_experiment:
            qtable_experiment.writelines(result)

