import numpy
from simulateNeuralNetwork import simulateNeuralNetwork

# Learning a single-layer network. The single-layer network is trained
# by the number of steps: learnSteps specified. Training of a single-layer
# network takes place on the given training string: trainStringInputs, trainStringOutputs

def learn(networkWeightsMatrixBeforeLearn,\
          trainStringInputs,\
          trainStringOutputs,\
          learnSteps,\
          maxLearnSteps,\
          supposedNetworkError):
    beta = 5
    dataPlot = {}
    learnFactor = 0.1
    minimumLearnSteps = 10
    sizeOfEachElementInsideArray = 1
    oneHalfMSEDivider = 2
    square = 2

    # Initialization of variables: examplesNumber, networkWeightsMatrix
    examplesNumber = trainStringInputs.shape[sizeOfEachElementInsideArray]
    networkWeightsMatrix = networkWeightsMatrixBeforeLearn

    for learnStep in range(1, learnSteps + 1):
        # Drawing an example.
        drawExample = numpy.random.randint(examplesNumber, size = 1)

        # Calculation of the output using the drawn input.
        inputExample = trainStringInputs[:, drawExample]
        result = simulateNeuralNetwork(networkWeightsMatrix, inputExample)

        # Calculating errors on outputs.
        resultError = trainStringOutputs[:, drawExample] - result
        resultMultipliedByDerivativeOfActivationFunction = resultError * beta * result * (1 - result)

        # Calculating the mean square error.
        # Preparation of data for the chart.
        meanSquaredError = numpy.sum(resultError ** square / oneHalfMSEDivider)
        dataPlot[learnStep] = meanSquaredError

        # Early termination of learning - Handler.
        # if - The number of steps cannot be smaller than: minimumLearnSteps.
        # Otherwise, the single-layer neural network may behave unstable.
        # Break when supposedNetworkError is reached.
        # elif - steps limit based on maxLearnSteps variable.
        if meanSquaredError <= supposedNetworkError and learnStep >= minimumLearnSteps:
            break
        elif learnStep >= maxLearnSteps:
            break

        # Calculations of weight corrections
        weightsAfterCorrection = learnFactor \
                                 * inputExample \
                                 * resultMultipliedByDerivativeOfActivationFunction.T

        # Use of corrected weight
        networkWeightsMatrix +=  weightsAfterCorrection

    return networkWeightsMatrix, dataPlot