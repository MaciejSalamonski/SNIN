import sys
import os
import numpy
import matplotlib.pyplot
from tabulate import tabulate
from matplotlib import cm
from createNetworkWithRandomWeights import createNetworkWithRandomWeights
from learn import learn
from simulateNeuralNetwork import simulateNeuralNetwork

trainStringInputs = numpy.array([[0, 0, 1, 1],
                                [0, 1, 0, 1]])
trainStringOutputs = numpy.array([[0, 1, 1, 0]])
learnSteps = 5000
maxLearnSteps = 2500
supposedNetworkError = 0.0003
numberOfShownExamplesPerStep = 10

maximumOfWeightRange = 100
minimumOfWeightRange = -100
noDataElementsLimitation = 2
square = 2
weightStepDenominator = 100

# calculateData() - Function responsible for two things: Creation, Learining neural network.

def calculateData():
    inputsNumber = 2 
    firstLayerNueronsNumber = 2 
    secondLayerNueronsNumbe = 1

    # Creation of a double-layer neural network.
    firstLayerNetworkWeightsMatrixBeforeLearn, \
    secondLayerNetworkWeightsMatrixBeforeLearn = createNetworkWithRandomWeights(inputsNumber, \
                                                                                firstLayerNueronsNumber, \
                                                                                secondLayerNueronsNumbe)

    # Learning the double-layer neural network.
    firstLayerNetworkWeightsMatrixAfterLearn, \
    secondLayerNetworkWeightsMatrixAfterLearn, \
    firstLayerDataPlot, \
    secondLayerDataPlot = learn(firstLayerNetworkWeightsMatrixBeforeLearn, \
                                secondLayerNetworkWeightsMatrixBeforeLearn, \
                                trainStringInputs, \
                                trainStringOutputs, \
                                learnSteps, \
                                maxLearnSteps, \
                                supposedNetworkError, \
                                numberOfShownExamplesPerStep)

    return firstLayerNetworkWeightsMatrixAfterLearn, \
           secondLayerNetworkWeightsMatrixAfterLearn, \
           firstLayerDataPlot, \
           secondLayerDataPlot

# Using a calculateData() function and assigning its result to variables.
firstLayerNetworkWeightsMatrixAfterLearn, \
secondLayerNetworkWeightsMatrixAfterLearn, \
firstLayerDataPlot, \
secondLayerDataPlot = calculateData()

# testDoubleNeuralNetwork() - Function responsible for testing neural network
# Obtaining the result of learning the double-layer neural network.
# The result will be used to test the neural network.

def testDoubleNeuralNetwork():
    firstWeightIndex = 0
    secondWeightIndex = 1
    thirdWeightIndex = 2
    fourthWeightIndex = 3

    firstLayerOutputsVector, \
    firstElementOfSecondLayerOutputsVector = simulateNeuralNetwork(firstLayerNetworkWeightsMatrixAfterLearn, \
                                                                   secondLayerNetworkWeightsMatrixAfterLearn, \
                                                                   trainStringInputs[:, [firstWeightIndex]])
    firstLayerOutputsVector, \
    secondElementOfSecondLayerOutputsVector = simulateNeuralNetwork(firstLayerNetworkWeightsMatrixAfterLearn, \
                                                                    secondLayerNetworkWeightsMatrixAfterLearn, \
                                                                    trainStringInputs[:, [secondWeightIndex]])
    firstLayerOutputsVector, \
    thirdElementOfSecondLayerOutputsVector = simulateNeuralNetwork(firstLayerNetworkWeightsMatrixAfterLearn, \
                                                                   secondLayerNetworkWeightsMatrixAfterLearn, \
                                                                   trainStringInputs[:, [thirdWeightIndex]])
    firstLayerOutputsVector, \
    fourthElementOfSecondLayerOutputsVector = simulateNeuralNetwork(firstLayerNetworkWeightsMatrixAfterLearn, \
                                                                    secondLayerNetworkWeightsMatrixAfterLearn, \
                                                                    trainStringInputs[:, [fourthWeightIndex]])
    SecondLayerOutputsVector = [firstElementOfSecondLayerOutputsVector, \
                                secondElementOfSecondLayerOutputsVector, \
                                thirdElementOfSecondLayerOutputsVector, \
                                fourthElementOfSecondLayerOutputsVector]

    stringOutputsAfterLearn = numpy.array([[]])
    for element in SecondLayerOutputsVector:
        stringOutputsAfterLearn = numpy.append(stringOutputsAfterLearn, element, axis = 1)

    return stringOutputsAfterLearn

# getResult() - Displaying the result in the form of a table for double-layer neural network. 

def getResult():
    # Using a testDoubleNeuralNetwork() function and assigning its result to stringOutputsAfterLearn variable.
    stringOutputsAfterLearn = testDoubleNeuralNetwork()
    print(tabulate(trainStringOutputs, \
                   tablefmt='orgtbl', \
                   showindex = ['Expected values'], \
                   headers = ['First Value', 'Second Value', 'Third Value', 'Fourth Value']))
    print()
    print(tabulate(stringOutputsAfterLearn, \
                   tablefmt='orgtbl', \
                   showindex = ['Values after learning'], \
                   headers = ['First Value', 'Second Value', 'Third Value', 'Fourth Value']))

# getMeanSquaredErrorChart() - Displaying the mean square error chart of double-layer neural network.

def getMeanSquaredErrorChart():
    figureColumns = 1
    figureRows = 2

    figure, meanSquaredErrorChart = matplotlib.pyplot.subplots(figureRows, figureColumns)

    meanSquaredErrorChart[0].set_title("Mean Squared Error - First Layer")
    meanSquaredErrorChart[0].set_ylabel('Error Value')
    meanSquaredErrorChart[0].set_xlabel('Learning Step')
    meanSquaredErrorChart[0].set_xlim(1, len(firstLayerDataPlot.keys()))
    meanSquaredErrorChart[0].grid()
    meanSquaredErrorChart[0].plot(list(firstLayerDataPlot.keys()), list(firstLayerDataPlot.values()))
    
    meanSquaredErrorChart[1].set_title("Mean Squared Error - Second Layer")
    meanSquaredErrorChart[1].set_ylabel('Error Value')
    meanSquaredErrorChart[1].set_xlabel('Learning Step')
    meanSquaredErrorChart[1].set_xlim(1, len(secondLayerDataPlot.keys()))
    meanSquaredErrorChart[1].grid()
    meanSquaredErrorChart[1].plot(list(secondLayerDataPlot.keys()), list(secondLayerDataPlot.values()))
    
    figure.tight_layout()

# getPurposeFunctionDependingOnWeightsChart() - Displaying purpose function depending on weight changes.

def getPurposeFunctionDependingOnWeightsChart():
    errorValues = []
    figureColumns = 1
    figureRows = 1
    weightIndex = 0

    # The range of the weight value change: (-1, 1) with 0.01 step
    # If calculating takes too long, You can change: weightStepDenominator value to 10.
    rangeOfWeightChange = [weightStep / weightStepDenominator for weightStep in range(minimumOfWeightRange, \
                                                                                      maximumOfWeightRange)]
    previousValueOfWeight = secondLayerNetworkWeightsMatrixAfterLearn[weightIndex]

    # Double-layer neural network testing for successive weight values
    for currentWeight in rangeOfWeightChange:
        secondLayerNetworkWeightsMatrixAfterLearn[weightIndex] = currentWeight
        
        stringOutputsAfterLearn = testDoubleNeuralNetwork()

        # Calculation of the mean square error for the current weight value
        expectedValueDeviation = trainStringOutputs - stringOutputsAfterLearn
        errorValues.append(numpy.sum(expectedValueDeviation ** square / noDataElementsLimitation))

    # Restore  original value after finished testing.
    secondLayerNetworkWeightsMatrixAfterLearn[weightIndex] = previousValueOfWeight

    figure, purposeFunctionDependingOnWeights = matplotlib.pyplot.subplots(figureRows, figureColumns)
    purposeFunctionDependingOnWeights.set_title("One weight change")
    purposeFunctionDependingOnWeights.set_ylabel('Error Value')
    purposeFunctionDependingOnWeights.set_xlabel('Weight Value')
    purposeFunctionDependingOnWeights.grid()
    purposeFunctionDependingOnWeights.plot(rangeOfWeightChange, errorValues)

# getPurposeFunctionDependingOnWeightsChart() - Displaying purpose function depending on the changes of both weights (3D).

def getPurposeFunctionDependingOnWeights3DChart():
    errorValues = []
    firstWeightIndex = 0
    secondWeightIndex = 1

    # The range of the weight value change: (-1, 1) with 0.01 step
    # If calculating takes too long, You can change: weightStepDenominator value to 10.
    rangeOfWeightChange = [weightStep / weightStepDenominator for weightStep in range(minimumOfWeightRange, \
                                                                                      maximumOfWeightRange)]
    previousValueOfFirstWeight = secondLayerNetworkWeightsMatrixAfterLearn[firstWeightIndex]
    previousValueOfSecondWeight = secondLayerNetworkWeightsMatrixAfterLearn[secondWeightIndex]

    # Double-layer neural network testing for successive weights values
    for index, currentValueOfFirstWeight in enumerate(rangeOfWeightChange):
        errorValues.append([])
        for currentValueOfSecondWeight in rangeOfWeightChange:
            secondLayerNetworkWeightsMatrixAfterLearn[firstWeightIndex] = currentValueOfFirstWeight
            secondLayerNetworkWeightsMatrixAfterLearn[secondWeightIndex] = currentValueOfSecondWeight
    
            stringOutputsAfterLearn = testDoubleNeuralNetwork()

            # Calculation of the mean square error for the current weight value
            expectedValueDeviation = trainStringOutputs - stringOutputsAfterLearn
            errorValues[index].append(numpy.sum(expectedValueDeviation ** square / noDataElementsLimitation))

    # Restore  original value after finished testing.
    secondLayerNetworkWeightsMatrixAfterLearn[firstWeightIndex] = previousValueOfFirstWeight
    secondLayerNetworkWeightsMatrixAfterLearn[secondWeightIndex] = previousValueOfSecondWeight

    figure = matplotlib.pyplot.figure()
    purposeFunctionDependingOnWeights = figure.add_subplot(projection='3d')
    purposeFunctionDependingOnWeights.set_title("Two weight change")
    purposeFunctionDependingOnWeights.set_xlabel("First Weight Value")
    purposeFunctionDependingOnWeights.set_ylabel("Second Weight Value")
    purposeFunctionDependingOnWeights.set_zlabel("Error Value")
    axisX, axisY = numpy.meshgrid(rangeOfWeightChange, rangeOfWeightChange)
    surface = purposeFunctionDependingOnWeights.plot_surface(axisX, \
                                                             axisY, \
                                                             numpy.array(errorValues), \
                                                             cmap = cm.RdBu)
    figure.colorbar(surface, shrink = 1, aspect = 10)
    figure.tight_layout()