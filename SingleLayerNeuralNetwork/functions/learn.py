import numpy
from simulateNeuralNetwork import *

def learn(networkWeightsMatrixBeforeLearn,\
          trainStringInputs,\
          trainStringOutputs,\
          learnSteps,\
          maxLearnSteps,\
          supposedNetworkError):
    beta = 5
    dataPlot = {}
    learnFactor = 0.1
    sizeOfEachElementInsideArray = 1

    examplesNumber = trainStringInputs.shape[sizeOfEachElementInsideArray]
    networkWeightsMatrix = networkWeightsMatrixBeforeLearn

    for learnStep in range(1, learnSteps + 1):
        drawExample = numpy.random.randint(examplesNumber, size = 1)
        inputExample = trainStringInputs[:, drawExample]

        result = simulateNeuralNetwork(networkWeightsMatrix, inputExample)
        resultError = trainStringOutputs[:, drawExample] - result
        resultMultipliedByDerivativeOfActivationFunction = resultError * beta * result * (1 - result)

        meanSquaredError = numpy.sum(resultError ** 2 / 2)
        dataPlot[learnStep] = meanSquaredError

        if meanSquaredError <= supposedNetworkError and learnStep >= 10:
            break
        elif learnStep >= maxLearnSteps:
            break

        weightsAfterCorrection = learnFactor * inputExample * resultMultipliedByDerivativeOfActivationFunction.T
        networkWeightsMatrix = networkWeightsMatrix + weightsAfterCorrection

    return networkWeightsMatrix, dataPlot