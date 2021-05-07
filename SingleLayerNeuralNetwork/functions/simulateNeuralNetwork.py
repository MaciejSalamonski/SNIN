import numpy

# Simulating the operation of a single-layer neural network.

def simulateNeuralNetwork(networkWeightsMatrix, inputsVector):
    beta = 5
    constantValue = 1

    weightedSum = networkWeightsMatrix.T.dot(inputsVector)
    outputsVector = (constantValue / (constantValue + numpy.exp(- beta * weightedSum)))
    
    return outputsVector