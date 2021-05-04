import numpy

def createNetworkWithRandomWeights(inputsNumber, nueronsNumber):
    fromMinusZeroPointOneToZeroPointOneRange = 0.1
    fromZeroToZeroPointTwoRange = 0.2

    return numpy.random.rand(inputsNumber, nueronsNumber)\
           * fromZeroToZeroPointTwoRange\
           - fromMinusZeroPointOneToZeroPointOneRange