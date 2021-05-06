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
maxLearnSteps = 3500
supposedNetworkError = 0.0003
numberOfShownExamplesPerStep = 10

def calculateData():
    inputsNumber = 2 
    firstLayerNueronsNumber = 2 
    secondLayerNueronsNumbe = 1

    firstLayerNetworkWeightsMatrixBeforeLearn, \
    secondLayerNetworkWeightsMatrixBeforeLearn = createNetworkWithRandomWeights(inputsNumber, \
                                                                                firstLayerNueronsNumber, \
                                                                                secondLayerNueronsNumbe)

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

firstLayerNetworkWeightsMatrixAfterLearn, \
secondLayerNetworkWeightsMatrixAfterLearn, \
firstLayerDataPlot, \
secondLayerDataPlot = calculateData()

def testDoubleNeuralNetwork():
    firstLayerOutputsVector, \
    firstElementOfSecondLayerOutputsVector = simulateNeuralNetwork(firstLayerNetworkWeightsMatrixAfterLearn, \
                                                                   secondLayerNetworkWeightsMatrixAfterLearn, \
                                                                   trainStringInputs[:, [0]])
    firstLayerOutputsVector, \
    secondElementOfSecondLayerOutputsVector = simulateNeuralNetwork(firstLayerNetworkWeightsMatrixAfterLearn, \
                                                                    secondLayerNetworkWeightsMatrixAfterLearn, \
                                                                    trainStringInputs[:, [1]])
    firstLayerOutputsVector, \
    thirdElementOfSecondLayerOutputsVector = simulateNeuralNetwork(firstLayerNetworkWeightsMatrixAfterLearn, \
                                                                   secondLayerNetworkWeightsMatrixAfterLearn, \
                                                                   trainStringInputs[:, [2]])
    firstLayerOutputsVector, \
    fourthElementOfSecondLayerOutputsVector = simulateNeuralNetwork(firstLayerNetworkWeightsMatrixAfterLearn, \
                                                                    secondLayerNetworkWeightsMatrixAfterLearn, \
                                                                    trainStringInputs[:, [3]])
    SecondLayerOutputsVector = [firstElementOfSecondLayerOutputsVector, \
                                secondElementOfSecondLayerOutputsVector, \
                                thirdElementOfSecondLayerOutputsVector, \
                                fourthElementOfSecondLayerOutputsVector]

    stringOutputsAfterLearn = numpy.array([[]])
    for element in SecondLayerOutputsVector:
        stringOutputsAfterLearn = numpy.append(stringOutputsAfterLearn, element, axis = 1)

    return stringOutputsAfterLearn

def getResult():
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

def wykresZmiennaWaga2D():
    blad = []
    zakres_zmiany_wagi = [x / 10 for x in range(-10, 11)]
    pierwotna_wartosc_wagi = secondLayerNetworkWeightsMatrixAfterLearn[0]

    for aktualna_wartosc_wagi in zakres_zmiany_wagi:
        secondLayerNetworkWeightsMatrixAfterLearn[0] = aktualna_wartosc_wagi
        
        stringOutputsAfterLearn = testDoubleNeuralNetwork()

        odchylenie_od_wart_oczekiwanej = trainStringOutputs - stringOutputsAfterLearn
        blad.append(numpy.sum(odchylenie_od_wart_oczekiwanej ** 2 / 2))

    secondLayerNetworkWeightsMatrixAfterLearn[0] = pierwotna_wartosc_wagi

    fig, ax = matplotlib.pyplot.subplots(1, 1)

    ax.plot(zakres_zmiany_wagi, blad)
    ax.grid()
    ax.set_title("Jedna zmienna waga")
    ax.set_ylabel('Wartość błędu')
    ax.set_xlabel('Wartość wagi')

def wykresZmiennaWaga3D():
    blad = []
    zakres_zmiany_wagi = [x / 10 for x in range(-10, 11)]
    pierwotna_wartosc_wagi1 = secondLayerNetworkWeightsMatrixAfterLearn[0]
    pierwotna_wartosc_wagi2 = secondLayerNetworkWeightsMatrixAfterLearn[1]

    for i, aktualna_wartosc_wagi1 in enumerate(zakres_zmiany_wagi):
        blad.append([])
        for aktualna_wartosc_wagi2 in zakres_zmiany_wagi:
            secondLayerNetworkWeightsMatrixAfterLearn[0] = aktualna_wartosc_wagi1
            secondLayerNetworkWeightsMatrixAfterLearn[1] = aktualna_wartosc_wagi2
    
            stringOutputsAfterLearn = testDoubleNeuralNetwork()

            odchylenie_od_wart_oczekiwanej = trainStringOutputs - stringOutputsAfterLearn
            blad[i].append(numpy.sum(odchylenie_od_wart_oczekiwanej ** 2 / 2))

    secondLayerNetworkWeightsMatrixAfterLearn[0] = pierwotna_wartosc_wagi1
    secondLayerNetworkWeightsMatrixAfterLearn[1] = pierwotna_wartosc_wagi2

    fig = matplotlib.pyplot.figure()

    ax = fig.gca(projection='3d')
    ax.set_title("Dwie zmienne wagi")
    ax.set_xlabel("Wartość wagi 1")
    ax.set_ylabel("Wartość wagi 2")
    ax.set_zlabel("Wartość błędu")
    X, Y = numpy.meshgrid(zakres_zmiany_wagi, zakres_zmiany_wagi)
    surf = ax.plot_surface(X, Y, numpy.array(blad), cmap=cm.seismic,
                    linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()