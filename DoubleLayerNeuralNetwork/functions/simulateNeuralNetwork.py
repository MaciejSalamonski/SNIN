import numpy

# simulateNeuralNetwork(args...) - Simulating the operation of a double-layer neural network.

def simulateNeuralNetwork(firstLayerNetworkWeightsMatrix, secondLayerNetworkWeightsMatrix, inputsVector):
    beta = 5
    constantValue = 1
    firstRowFilledWithMinusOne = -1

    firstLayerInputVector = numpy.vstack((firstRowFilledWithMinusOne, inputsVector))
    firstLayerWeightedSum = firstLayerNetworkWeightsMatrix.T.dot(firstLayerInputVector)
    firstLayerOutputsVector = (constantValue / (constantValue + numpy.exp(- beta * firstLayerWeightedSum)))

    secondLayerInputVector = numpy.vstack((firstRowFilledWithMinusOne, firstLayerOutputsVector))
    secondLayerWeightedSum = secondLayerNetworkWeightsMatrix.T.dot(secondLayerInputVector)
    secondLayerOutputsVector = (constantValue / (constantValue + numpy.exp(- beta * secondLayerWeightedSum)))

    return firstLayerOutputsVector, secondLayerOutputsVector