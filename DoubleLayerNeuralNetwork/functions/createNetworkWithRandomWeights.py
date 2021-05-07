import numpy

# createNetworkWithRandomWeights(args...) - Creating a double-layer neural network.
# Filling both weight matrices with random values in the range: (-0.1, 0.1).

def createNetworkWithRandomWeights(inputsNumber, firstLayerNueronsNumber, secondLayerNueronsNumber):
    fromMinusZeroPointOneToZeroPointOneRange = 0.1
    fromZeroToZeroPointTwoRange = 0.2

    return numpy.random.rand(inputsNumber + 1, firstLayerNueronsNumber) \
           * fromZeroToZeroPointTwoRange \
           - fromMinusZeroPointOneToZeroPointOneRange, \
           numpy.random.rand(firstLayerNueronsNumber + 1, secondLayerNueronsNumber) \
           * fromZeroToZeroPointTwoRange \
           - fromMinusZeroPointOneToZeroPointOneRange