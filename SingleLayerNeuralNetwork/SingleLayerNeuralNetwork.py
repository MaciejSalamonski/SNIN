import matplotlib.pyplot
import numpy
import sys
import os
sys.path.append(os.path.abspath("functions"))
from createNetworkWithRandomWeights import *
from learn import *
from simulateNeuralNetwork import *
from tabulate import tabulate

if __name__ == '__main__':

    inputsNumber = 5
    learnSteps = 70
    maxLearnSteps = 40
    nueronsNumber = 3
    supposedNetworkError = 0.0002
    trainStringInputs = numpy.array([[4, 2, -1],
                                [0.01, -1, 3.5],
                                [0.01, 2, 0.01],
                                [-1, 2.5, -2],
                                [-1.5, 2, 1.5]])
    trainStringOutputs = numpy.eye(3)

    networkWeightsMatrixBeforeLearn = createNetworkWithRandomWeights(inputsNumber, nueronsNumber)
    networkWeightsMatrixAfterLearn, dataPlot = learn(networkWeightsMatrixBeforeLearn,\
                           trainStringInputs,\
                           trainStringOutputs,\
                           learnSteps,\
                           maxLearnSteps,\
                           supposedNetworkError)
    result = simulateNeuralNetwork(networkWeightsMatrixAfterLearn, trainStringInputs)

    print(tabulate(result,\
                   tablefmt = 'orgtbl',\
                   showindex = ['Mammal', 'Bird', 'Fish'],\
                   headers = ['First Example', 'Second Example', 'Third Example']))

    figure, meanSquaredErrorChart = matplotlib.pyplot.subplots(1, 1)
    meanSquaredErrorChart.set_title("Total mean squared error")
    meanSquaredErrorChart.set_xlabel('Learning Step')
    meanSquaredErrorChart.set_ylabel('Error Value')
    meanSquaredErrorChart.grid()
    meanSquaredErrorChart.plot(list(dataPlot.keys()), list(dataPlot.values()))
    meanSquaredErrorChart.set_xlim(1, len(dataPlot.keys()))
    matplotlib.pyplot.show()