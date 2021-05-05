import matplotlib.pyplot
import numpy

from createNetworkWithRandomWeights import createNetworkWithRandomWeights
from learn import learn
from simulateNeuralNetwork import simulateNeuralNetwork
from tabulate import tabulate

inputsNumber = 5
learnSteps = 70
maxLearnSteps = 40
nueronsNumber = 3
sizeOfAnIdentityMatrix = 3
supposedNetworkError = 0.0002
trainStringInputs = numpy.array([[4, 2, -1],
                                [0.01, -1, 3.5],
                                [0.01, 2, 0.01],
                                [-1, 2.5, -2],
                                [-1.5, 2, 1.5]])
trainStringOutputs = numpy.eye(sizeOfAnIdentityMatrix)

def calculateData():
    networkWeightsMatrixBeforeLearn = createNetworkWithRandomWeights(inputsNumber, nueronsNumber)
    networkWeightsMatrixAfterLearn, dataPlot = learn(networkWeightsMatrixBeforeLearn,\
                                                     trainStringInputs,\
                                                     trainStringOutputs,\
                                                     learnSteps,\
                                                     maxLearnSteps,\
                                                     supposedNetworkError)
    result = simulateNeuralNetwork(networkWeightsMatrixAfterLearn, trainStringInputs)

    return result, dataPlot

result, dataPlot = calculateData()

def getResult():
    print(tabulate(result,\
                tablefmt = 'orgtbl',\
                showindex = ['Mammal', 'Bird', 'Fish'],\
                headers = ['First Example', 'Second Example', 'Third Example']))

def getMeanSquaredErrorChart():
    figureColumns = 1
    figureRows = 1
    figure, meanSquaredErrorChart = matplotlib.pyplot.subplots(figureRows, figureColumns)
    meanSquaredErrorChart.set_title("Total mean squared error")
    meanSquaredErrorChart.set_xlabel('Learning Step')
    meanSquaredErrorChart.set_ylabel('Error Value')
    meanSquaredErrorChart.grid()
    meanSquaredErrorChart.plot(list(dataPlot.keys()), list(dataPlot.values()))
    meanSquaredErrorChart.set_xlim(1, len(dataPlot.keys()))
    matplotlib.pyplot.show()