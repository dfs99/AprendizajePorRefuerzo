#!bin/bash
echo "comenzando entrenamiento..."
for mapa in labAA1 labAA2 labAA3 labAA4 labAA5
do
    python busters.py -n 3000 -l $mapa -p QLearningAgent -g RandomGhost -t 0 -k 1 -e ExperimetoGlobal
done
echo "Entrenamiento Finalizado"