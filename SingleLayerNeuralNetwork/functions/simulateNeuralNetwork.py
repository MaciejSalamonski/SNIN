import numpy

def simulateNeuralNetwork(networkWeightsMatrix, inputsVector):
    beta = 5
    constantValue = 1

    outputsVector = networkWeightsMatrix.T.dot(inputsVector)
    
    return (constantValue / (constantValue + numpy.exp(- beta * outputsVector)))