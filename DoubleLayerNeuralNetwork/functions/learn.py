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

    beta = 5
    firstLayerDataPlot = {}
    firstLayerMeanSquaredError = 0
    firstLayerWeightsAfterCorrection = 0
    learnFactor = 0.1
    momentumFactor = 0.7
    oneHalfMSEDivider = 2
    secondLayerDataPlot = {}
    secondLayerMeanSquaredError = 0
    secondLayerWeightsAfterCorrection = 0
    sizeOfEachElementInsideArray = 1
    square = 2

    blad2poprzedni = 0
    dW1pokaz = 0
    dW2pokaz = 0

    firstLayerNetworkWeightsMatrix = firstLayerNetworkWeightsMatrixBeforeLearn
    secondLayerNetworkWeightsMatrix = secondLayerNetworkWeightsMatrixBeforeLearn

    S2 = secondLayerNetworkWeightsMatrix.shape[0]

    examplesNumber = trainStringInputs.shape[sizeOfEachElementInsideArray]

    for learnStep in range(1, learnSteps + 1):
        for krok_pokazu in range(numberOfShownExamplesPerStep):
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

            firstLayerResultError = secondLayerNetworkWeightsMatrix[1:S2, :] * secondLayerResultMultipliedByDerivativeOfActivationFunction
            firstLayerResultMultipliedByDerivativeOfActivationFunction = beta \
                                                                         * firstLayerResultError \
                                                                         * firstLayerResult \
                                                                         * (1 - firstLayerResult)

            firstLayerMeanSquaredError += numpy.sum(firstLayerResultError ** square / oneHalfMSEDivider)
            secondLayerMeanSquaredError += numpy.sum(secondLayerResultError ** square / oneHalfMSEDivider)

            X1 = numpy.vstack((-1, inputExample))
            firstLayerWeightsAfterCorrection = learnFactor \
                                               * X1 \
                                               * firstLayerResultMultipliedByDerivativeOfActivationFunction.T \
                                               + (momentumFactor * firstLayerWeightsAfterCorrection)
            X2 = numpy.vstack((-1, firstLayerResult))
            secondLayerWeightsAfterCorrection = learnFactor \
                                                * X2 \
                                                * secondLayerResultMultipliedByDerivativeOfActivationFunction.T \
                                                + (momentumFactor * secondLayerWeightsAfterCorrection)

            dW1pokaz += firstLayerWeightsAfterCorrection 
            dW2pokaz += secondLayerWeightsAfterCorrection

        firstLayerNetworkWeightsMatrix += dW1pokaz / numberOfShownExamplesPerStep
        secondLayerNetworkWeightsMatrix += dW2pokaz / numberOfShownExamplesPerStep

        dW1pokaz = 0
        dW2pokaz = 0

        blad1 = firstLayerMeanSquaredError / numberOfShownExamplesPerStep
        blad2 = secondLayerMeanSquaredError / numberOfShownExamplesPerStep
        secondLayerDataPlot[learnStep] = blad2
        firstLayerDataPlot[learnStep] = blad1

        firstLayerMeanSquaredError = 0
        secondLayerMeanSquaredError = 0

        if learnStep >= maxLearnSteps:
            break
        elif blad2 <= supposedNetworkError:
            try:
                if any(list(secondLayerDataPlot.values())[learnStep - b] / supposedNetworkError >= 10 for b in range(2, 42)):
                    pass
                else:
                    break
            except IndexError:
                pass

        if blad2 > 1.04 * blad2poprzedni and 0.7 * learnFactor >= 0.15:
            learnFactor = 0.7 * learnFactor
        else:
            learnFactor = 1.05 * learnFactor
        blad2poprzedni = blad2

    return firstLayerNetworkWeightsMatrix, \
           secondLayerNetworkWeightsMatrix, \
           firstLayerDataPlot, \
           secondLayerDataPlot