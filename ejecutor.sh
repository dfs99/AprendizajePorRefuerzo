#!bin/bash
echo "Empezando experimento..."
echo "Se empieza a ejecutar con fantasmas aleatorios."
python busters.py -n 5000 -l labAA5 -p QLearningAgent -g RandomGhost -t 0 -k 3 -q -e Walls_refined_dinamico
echo "Se empieza a ejecutar sobre fantasmas estáticos."
python busters.py -n 5000 -l labAA5 -p QLearningAgent -t 0 -k 3 -q -e Walls_refined_estatico
echo "fin esperimentos"