import sys
import os
import numpy
import matplotlib.pyplot
from tabulate import tabulate
from matplotlib import cm
from createNetworkWithRandomWeights import createNetworkWithRandomWeights
from learn import learn
from simulateNeuralNetwork import simulateNeuralNetwork

P = numpy.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
T = numpy.array([[0, 1, 1, 0]])
ilosc_petli_nauczania = 5000
maks_ilosc_krokow_nauczania = 3500
blad_do_osiagniecia = 0.0003
liczba_pokazow = 10

def calculateData():
    W1przed, W2przed = createNetworkWithRandomWeights(2, 2, 1)
    W1po, W2po, plot_data1, plot_data2 = learn(W1przed, W2przed, P, T, ilosc_petli_nauczania, maks_ilosc_krokow_nauczania, blad_do_osiagniecia, liczba_pokazow)

    return W1po, W2po, plot_data1, plot_data2

W1po, W2po, plot_data1, plot_data2 = calculateData()

def testujSiec():
    Y1, Y2a = simulateNeuralNetwork(W1po, W2po, P[:, [0]])
    Y1, Y2b = simulateNeuralNetwork(W1po, W2po, P[:, [1]])
    Y1, Y2c = simulateNeuralNetwork(W1po, W2po, P[:, [2]])
    Y1, Y2d = simulateNeuralNetwork(W1po, W2po, P[:, [3]])
    Y = [Y2a, Y2b, Y2c, Y2d]

    Ypo = numpy.array([[]])
    for i in Y:
        Ypo = numpy.append(Ypo, i, axis=1)

    return Ypo

def pokazEfektNaukiSieci():
    Ypo = testujSiec()
    print("Wartosci oczekiwane:\n", tabulate(T, tablefmt='fancy_grid'), sep='')
    print("Wyniki po nauczniu sieci:\n", tabulate(Ypo, tablefmt='fancy_grid'), sep='')

def wykresBleduSredniokwadratowego():
    fig, ax = matplotlib.pyplot.subplots(2, 1)

    ax[0].plot(list(plot_data1.keys()), list(plot_data1.values()))
    ax[0].set_xlim(1, len(plot_data1.keys()))
    ax[0].grid()
    ax[0].set_title("MSE warstwa 1")
    ax[0].set_ylabel('Wartość błędu')
    ax[0].set_xlabel('Krok uczenia')

    ax[1].plot(list(plot_data2.keys()), list(plot_data2.values()))
    ax[1].set_xlim(1, len(plot_data2.keys()))
    ax[1].grid()
    ax[1].set_title("MSE warstwa 2")
    ax[1].set_ylabel('Wartość błędu')
    ax[1].set_xlabel('Krok uczenia')

    fig.tight_layout()

def wykresZmiennaWaga2D():
    blad = []
    zakres_zmiany_wagi = [x / 10 for x in range(-10, 11)]
    pierwotna_wartosc_wagi = W2po[0]

    for aktualna_wartosc_wagi in zakres_zmiany_wagi:
        W2po[0] = aktualna_wartosc_wagi
        
        Ypo = testujSiec()

        odchylenie_od_wart_oczekiwanej = T - Ypo
        blad.append(numpy.sum(odchylenie_od_wart_oczekiwanej ** 2 / 2))

    W2po[0] = pierwotna_wartosc_wagi

    fig, ax = matplotlib.pyplot.subplots(1, 1)

    ax.plot(zakres_zmiany_wagi, blad)
    ax.grid()
    ax.set_title("Jedna zmienna waga")
    ax.set_ylabel('Wartość błędu')
    ax.set_xlabel('Wartość wagi')

def wykresZmiennaWaga3D():
    blad = []
    zakres_zmiany_wagi = [x / 10 for x in range(-10, 11)]
    pierwotna_wartosc_wagi1 = W2po[0]
    pierwotna_wartosc_wagi2 = W2po[1]

    for i, aktualna_wartosc_wagi1 in enumerate(zakres_zmiany_wagi):
        blad.append([])
        for aktualna_wartosc_wagi2 in zakres_zmiany_wagi:
            W2po[0] = aktualna_wartosc_wagi1
            W2po[1] = aktualna_wartosc_wagi2
    
            Ypo = testujSiec()

            odchylenie_od_wart_oczekiwanej = T - Ypo
            blad[i].append(numpy.sum(odchylenie_od_wart_oczekiwanej ** 2 / 2))

    W2po[0] = pierwotna_wartosc_wagi1
    W2po[1] = pierwotna_wartosc_wagi2

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