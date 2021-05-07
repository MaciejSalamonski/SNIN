import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("functions"))
from calculateData import getResult
from calculateData import getMeanSquaredErrorChart
from calculateData import getPurposeFunctionDependingOnWeightsChart
from calculateData import getPurposeFunctionDependingOnWeights3DChart

if __name__ == '__main__':

    # Main - Calling getResult() & getMeanSquaredErrorChart() 
    # getPurposeFunctionDependingOnWeightsChart() & getPurposeFunctionDependingOnWeights3DChart()
    
    getResult()
    getMeanSquaredErrorChart()
    getPurposeFunctionDependingOnWeightsChart()
    getPurposeFunctionDependingOnWeights3DChart()
    
    # Displaying charts
    plt.show()
