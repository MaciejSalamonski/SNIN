import numpy

def simulateNeuralNetwork(networkWeightsMatrix, inputsVector):
    beta = 5
    constantValue = 1

    weightedSum = networkWeightsMatrix.T.dot(inputsVector)
    outputsVector = (constantValue / (constantValue + numpy.exp(- beta * weightedSum)))
    
    return outputsVector