import numpy

def simulateNeuralNetwork(firstLayerNetworkWeightsMatrix, secondLayerNetworkWeightsMatrix, inputsVector):
    beta = 5
    constantValue = 1

    firstLayerInputVector = numpy.vstack((-1, inputsVector))
    U1 = firstLayerNetworkWeightsMatrix.T.dot(firstLayerInputVector)
    firstLayerOutputsVector = (constantValue / (constantValue + numpy.exp(- beta * U1)))

    secondLayerInputVector = numpy.vstack((-1, firstLayerOutputsVector))
    U2 = secondLayerNetworkWeightsMatrix.T.dot(secondLayerInputVector)
    secondLayerOutputsVector = (constantValue / (constantValue + numpy.exp(- beta * U2)))

    return firstLayerOutputsVector, secondLayerOutputsVector