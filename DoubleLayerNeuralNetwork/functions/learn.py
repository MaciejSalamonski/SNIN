import numpy 
from simulateNeuralNetwork import *

def learn(W1przed, W2przed, P, T, n, m, e, k):
    liczbaPrzykladow = P.shape[1]

    W1 = W1przed
    W2 = W2przed

    S2 = W2.shape[0]

    wspMomentum = 0.7
    wspUcz = 0.1
    blad2poprzedni = 0
    dW1 = 0
    dW2 = 0
    beta = 5
    plot_data2 = {}
    plot_data1 = {}
    dW1pokaz = 0
    dW2pokaz = 0
    blad1pokaz = 0
    blad2pokaz = 0

    for krok_uczenia in range(1, n + 1):
        for krok_pokazu in range(k):
            nrPrzykladu = numpy.random.randint(liczbaPrzykladow, size=1)

            X = P[:, nrPrzykladu]
            X1 = numpy.vstack((-1, X))
            Y1, Y2 = simulateNeuralNetwork(W1, W2, X)

            X2 = numpy.vstack((-1, Y1))

            D2 = T[:, nrPrzykladu] - Y2
            E2 = beta * D2  * Y2 * (1 - Y2)

            D1 = W2[1:S2, :] * E2
            E1 = beta * D1  * Y1 * (1 - Y1)

            blad1pokaz += numpy.sum(D1 ** 2 / 2)
            blad2pokaz += numpy.sum(D2 ** 2 / 2)

            dW1 = wspUcz * X1 * E1.T + wspMomentum * dW1
            dW2 = wspUcz * X2 * E2.T + wspMomentum * dW2

            dW1pokaz += dW1 
            dW2pokaz += dW2

        W1 += dW1pokaz / k
        W2 += dW2pokaz / k

        dW1pokaz = 0
        dW2pokaz = 0

        blad1 = blad1pokaz / k
        blad2 = blad2pokaz / k
        plot_data2[krok_uczenia] = blad2
        plot_data1[krok_uczenia] = blad1

        blad1pokaz = 0
        blad2pokaz = 0

        if krok_uczenia >= m:
            break
        elif blad2 <= e:
            try:
                if any(list(plot_data2.values())[krok_uczenia - b] / e >= 10 for b in range(2, 42)):
                    pass
                else:
                    break
            except IndexError:
                pass

        if blad2 > 1.04 * blad2poprzedni and 0.7 * wspUcz >= 0.15:
            wspUcz = 0.7 * wspUcz
        else:
            wspUcz = 1.05 * wspUcz
        blad2poprzedni = blad2

    return W1, W2, plot_data1, plot_data2