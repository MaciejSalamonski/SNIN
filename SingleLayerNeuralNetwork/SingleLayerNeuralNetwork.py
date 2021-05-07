import numpy
import os
import sys

sys.path.append(os.path.abspath("functions"))
from calculateData import getMeanSquaredErrorChart
from calculateData import getResult

if __name__ == '__main__':

    # Main - Calling getResult() & getMeanSquaredErrorChart() function

    getResult()
    getMeanSquaredErrorChart()