import numpy

# Creating a single-layer neural network.
# Filling the weight matrix with random values in the range: (-0.1, 0.1).

def createNetworkWithRandomWeights(inputsNumber, nueronsNumber):
    fromMinusZeroPointOneToZeroPointOneRange = 0.1
    fromZeroToZeroPointTwoRange = 0.2

    return numpy.random.rand(inputsNumber, nueronsNumber)\
           * fromZeroToZeroPointTwoRange\
           - fromMinusZeroPointOneToZeroPointOneRange