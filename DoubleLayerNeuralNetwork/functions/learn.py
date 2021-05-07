import numpy 
from simulateNeuralNetwork import simulateNeuralNetwork

def learn(firstLayerNetworkWeightsMatrixBeforeLearn, \
          secondLayerNetworkWeightsMatrixBeforeLearn, \
          trainStringInputs, \
          trainStringOutputs, \
          learnSteps, \
          maxLearnSteps, 
          supposedNetworkError, \
          numberOfShownExamplesPerStep):

    assumedRatio = 10
    beta = 5
    firstLayerDataPlot = {}
    firstLayerMeanSquaredError = 0
    firstLayerWeightsAfterCorrection = 0
    learnFactor = 0.1
    maximumPreviousValueRange = 42
    minimumPreviousValueRange = 2
    momentumFactor = 0.7
    oneHalfMSEDivider = 2
    previousSecondLayerAveragedMeanSquaredError = 0
    secondLayerDataPlot = {}
    secondLayerMeanSquaredError = 0
    secondLayerWeightsAfterCorrection = 0
    showFirstLayerWeightsAfterCorrection = 0
    showSecondLayerWeightsAfterCorrection = 0
    sizeOfEachElementInsideArray = 1
    square = 2

    firstLayerNetworkWeightsMatrix = firstLayerNetworkWeightsMatrixBeforeLearn
    secondLayerNetworkWeightsMatrix = secondLayerNetworkWeightsMatrixBeforeLearn

    secondLayerInputsNumber = secondLayerNetworkWeightsMatrix.shape[0]
    examplesNumber = trainStringInputs.shape[sizeOfEachElementInsideArray]

    for learnStep in range(1, learnSteps + 1):
        for showStep in range(numberOfShownExamplesPerStep):
            drawExample = numpy.random.randint(examplesNumber, size = 1)
            inputExample = trainStringInputs[:, drawExample]
            
            firstLayerResult, secondLayerResult = simulateNeuralNetwork(firstLayerNetworkWeightsMatrix, \
                                                                        secondLayerNetworkWeightsMatrix, \
                                                                        inputExample)

            secondLayerResultError = trainStringOutputs[:, drawExample] - secondLayerResult
            secondLayerResultMultipliedByDerivativeOfActivationFunction = beta \
                                                                          * secondLayerResultError \
                                                                          * secondLayerResult \
                                                                          * (1 - secondLayerResult)

            firstLayerResultError = secondLayerNetworkWeightsMatrix[1:secondLayerInputsNumber, :] \
                                    * secondLayerResultMultipliedByDerivativeOfActivationFunction
            firstLayerResultMultipliedByDerivativeOfActivationFunction = beta \
                                                                         * firstLayerResultError \
                                                                         * firstLayerResult \
                                                                         * (1 - firstLayerResult)

            firstLayerMeanSquaredError += numpy.sum(firstLayerResultError ** square / oneHalfMSEDivider)
            secondLayerMeanSquaredError += numpy.sum(secondLayerResultError ** square / oneHalfMSEDivider)

            firstLayerInput = numpy.vstack((-1, inputExample))
            firstLayerWeightsAfterCorrection = learnFactor \
                                               * firstLayerInput \
                                               * firstLayerResultMultipliedByDerivativeOfActivationFunction.T \
                                               + (momentumFactor * firstLayerWeightsAfterCorrection)
            secondLayerInput = numpy.vstack((-1, firstLayerResult))
            secondLayerWeightsAfterCorrection = learnFactor \
                                                * secondLayerInput \
                                                * secondLayerResultMultipliedByDerivativeOfActivationFunction.T \
                                                + (momentumFactor * secondLayerWeightsAfterCorrection)

            showFirstLayerWeightsAfterCorrection += firstLayerWeightsAfterCorrection 
            showSecondLayerWeightsAfterCorrection += secondLayerWeightsAfterCorrection

        firstLayerNetworkWeightsMatrix += showFirstLayerWeightsAfterCorrection / numberOfShownExamplesPerStep
        secondLayerNetworkWeightsMatrix += showSecondLayerWeightsAfterCorrection / numberOfShownExamplesPerStep

        showFirstLayerWeightsAfterCorrection = 0
        showSecondLayerWeightsAfterCorrection = 0

        firstLayerAveragedMeanSquaredError = firstLayerMeanSquaredError / numberOfShownExamplesPerStep
        secondLayerAveragedMeanSquaredError = secondLayerMeanSquaredError / numberOfShownExamplesPerStep

        firstLayerDataPlot[learnStep] = firstLayerAveragedMeanSquaredError
        secondLayerDataPlot[learnStep] = secondLayerAveragedMeanSquaredError

        firstLayerMeanSquaredError = 0
        secondLayerMeanSquaredError = 0

        if learnStep >= maxLearnSteps:
            break
        elif secondLayerAveragedMeanSquaredError <= supposedNetworkError:
            try:
                if any(list(secondLayerDataPlot.values())[learnStep - previousValueIndex] / supposedNetworkError \
                            >= assumedRatio for previousValueIndex in range(minimumPreviousValueRange, \
                                                                            maximumPreviousValueRange)):
                    pass
                else:
                    break
            except IndexError:
                pass

        if secondLayerAveragedMeanSquaredError > 1.04 * previousSecondLayerAveragedMeanSquaredError and 0.7 * learnFactor >= 0.15:
            learnFactor = 0.7 * learnFactor
        else:
            learnFactor = 1.05 * learnFactor

        previousSecondLayerAveragedMeanSquaredError = secondLayerAveragedMeanSquaredError

    return firstLayerNetworkWeightsMatrix, \
           secondLayerNetworkWeightsMatrix, \
           firstLayerDataPlot, \
           secondLayerDataPlot