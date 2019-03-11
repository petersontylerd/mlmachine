
import numpy as np
import pandas as pd

def missingDataDropperAll(self):
    """

    """
    missingCols = self.X_.isnull().sum()
    missingCols = missingCols[missingCols > 0].index
    self.X_ = self.X_.drop(missingCols, axis = 1)

def featureDropper(self, cols):
    self.X_ = self.X_.drop(cols, axis = 1)

def missingColCompare(self, train, test):
    trainMissing = train.isnull().sum()
    trainMissing = trainMissing[trainMissing > 0].index

    testMissing = test.isnull().sum()
    testMissing = testMissing[testMissing > 0].index

    print('Missing in train, not test')
    print(set(testMissing) - set(trainMissing))
    print('')
    print('Missing in test, not train')
    print(set(trainMissing) - set(testMissing))