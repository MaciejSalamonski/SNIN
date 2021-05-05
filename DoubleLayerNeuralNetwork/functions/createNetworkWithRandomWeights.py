import numpy

def createNetworkWithRandomWeights(inputsNumber, firstLayerNueronsNumber, secondLayerNueronsNumber):
    fromMinusZeroPointOneToZeroPointOneRange = 0.1
    fromZeroToZeroPointTwoRange = 0.2

    return numpy.random.rand(inputsNumber + 1, firstLayerNueronsNumber) \
           * fromZeroToZeroPointTwoRange \
           - fromMinusZeroPointOneToZeroPointOneRange, \
           numpy.random.rand(firstLayerNueronsNumber + 1, secondLayerNueronsNumber) \
           * fromZeroToZeroPointTwoRange \
           - fromMinusZeroPointOneToZeroPointOneRange