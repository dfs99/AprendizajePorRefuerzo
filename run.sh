#!bin/bash
times=10
for map in bigHunt 20Hunt classic oneHunt openHunt
do
    touch "./logs/training_tutorial1".arff
    python busters.py -n $times -l $map -p BasicAgentAA -g RandomGhost -t 0 > "./logs/training_tutorial1".arff
done

times=5
for map in bigHunt 20Hunt classic oneHunt openHunt
do
    touch "./logs/test_samemaps_tutorial1".arff
    python busters.py -n $times -l $map -p BasicAgentAA -g RandomGhost -t 0 > "./logs/test_samemaps_tutorial1".arff
done

times=5
for map in smallHunt newmap mimapa mapa2 mapa3
do
    touch "./logs/test_othermaps_tutorial1".arff
    python busters.py -n $times -l $map -p BasicAgentAA -g RandomGhost -t 0 > "./logs/test_othermaps_tutorial1".arff
done

#python3 busters.py -n 1 -l smallHunt -p BasicAgentAA -g RandomGhost -t 0
