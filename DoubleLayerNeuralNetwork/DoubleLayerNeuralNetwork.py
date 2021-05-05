import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("functions"))
from calculateData import pokazEfektNaukiSieci
from calculateData import wykresBleduSredniokwadratowego
from calculateData import wykresZmiennaWaga2D
from calculateData import wykresZmiennaWaga3D

if __name__ == '__main__':
    
    pokazEfektNaukiSieci()
    wykresBleduSredniokwadratowego()
    wykresZmiennaWaga2D()
    wykresZmiennaWaga3D()
    plt.show()
