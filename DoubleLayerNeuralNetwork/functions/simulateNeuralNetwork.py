import numpy

def simulateNeuralNetwork(firstLayerNetworkWeightsMatrix, secondLayerNetworkWeightsMatrix, inputsVector):
    beta = 5
    constantValue = 1

    firstLayerInputVector = numpy.vstack((-1, inputsVector))
    firstLayerWeightedSum = firstLayerNetworkWeightsMatrix.T.dot(firstLayerInputVector)
    firstLayerOutputsVector = (constantValue / (constantValue + numpy.exp(- beta * firstLayerWeightedSum)))

    secondLayerInputVector = numpy.vstack((-1, firstLayerOutputsVector))
    secondLayerWeightedSum = secondLayerNetworkWeightsMatrix.T.dot(secondLayerInputVector)
    secondLayerOutputsVector = (constantValue / (constantValue + numpy.exp(- beta * secondLayerWeightedSum)))

    return firstLayerOutputsVector, secondLayerOutputsVector