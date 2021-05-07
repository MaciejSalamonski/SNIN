import numpy 
from simulateNeuralNetwork import simulateNeuralNetwork

# learn(args...) - Learning a double-layer network. The double-layer network is trained
# by the number of steps: learnSteps specified. Training of a double-layer
# network takes place on the given training string: trainStringInputs, trainStringOutputs

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

    # Initialization of variables: secondLayerInputsNumber, examplesNumber
    secondLayerInputsNumber = secondLayerNetworkWeightsMatrix.shape[0]
    examplesNumber = trainStringInputs.shape[sizeOfEachElementInsideArray]

    for learnStep in range(1, learnSteps + 1):
        for showStep in range(numberOfShownExamplesPerStep):
            # Drawing an example.
            drawExample = numpy.random.randint(examplesNumber, size = 1)

            # Calculation of the output using the drawn input.
            inputExample = trainStringInputs[:, drawExample]
            firstLayerResult, secondLayerResult = simulateNeuralNetwork(firstLayerNetworkWeightsMatrix, \
                                                                        secondLayerNetworkWeightsMatrix, \
                                                                        inputExample)

            # Calculating errors on outputs.
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

            # Calculating the mean square error.
            firstLayerMeanSquaredError += numpy.sum(firstLayerResultError ** square / oneHalfMSEDivider)
            secondLayerMeanSquaredError += numpy.sum(secondLayerResultError ** square / oneHalfMSEDivider)

            # Calculation of weight corrections. Application of momentum.
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

        # Use of corrected weight
        firstLayerNetworkWeightsMatrix += showFirstLayerWeightsAfterCorrection / numberOfShownExamplesPerStep
        secondLayerNetworkWeightsMatrix += showSecondLayerWeightsAfterCorrection / numberOfShownExamplesPerStep

        # Reset of temporary values
        showFirstLayerWeightsAfterCorrection = 0
        showSecondLayerWeightsAfterCorrection = 0

        # Averaging the mean square error obtained during the (show?) and saving it. Reset of auxiliary variables.
        firstLayerAveragedMeanSquaredError = firstLayerMeanSquaredError / numberOfShownExamplesPerStep
        secondLayerAveragedMeanSquaredError = secondLayerMeanSquaredError / numberOfShownExamplesPerStep

        firstLayerDataPlot[learnStep] = firstLayerAveragedMeanSquaredError
        secondLayerDataPlot[learnStep] = secondLayerAveragedMeanSquaredError

        firstLayerMeanSquaredError = 0
        secondLayerMeanSquaredError = 0

        # Early termination of learning - Handler.
        # if - Steps limit based on maxLearnSteps variable.
        # elif - Break when supposedNetworkError is reached.
        if learnStep >= maxLearnSteps:
            break
        elif secondLayerAveragedMeanSquaredError <= supposedNetworkError:
            try:
                # We assume that the previous error values ​​must satisfy below condition. 
                if any(list(secondLayerDataPlot.values())[learnStep - previousValueIndex] / supposedNetworkError \
                            >= assumedRatio for previousValueIndex in range(minimumPreviousValueRange, \
                                                                            maximumPreviousValueRange)):
                    pass
                else:
                    break
            # To avoid mistakes in the early stages of learning. 
            # There is a risk of index_out_of_range exception when the dictionary with errors is small.
            except IndexError:
                pass

        # Adaptive learning factor.
        if secondLayerAveragedMeanSquaredError > 1.04 * previousSecondLayerAveragedMeanSquaredError and 0.7 * learnFactor >= 0.15:
            learnFactor = 0.7 * learnFactor
        else:
            learnFactor = 1.05 * learnFactor

        previousSecondLayerAveragedMeanSquaredError = secondLayerAveragedMeanSquaredError

    return firstLayerNetworkWeightsMatrix, \
           secondLayerNetworkWeightsMatrix, \
           firstLayerDataPlot, \
           secondLayerDataPlot