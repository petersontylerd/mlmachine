
import numpy as np
import pandas as pd

def missingDataDropperAll(self):
    """

    """
    missingCols = self.data.isnull().sum()
    missingCols = missingCols[missingCols > 0].index
    self.data = self.data.drop(missingCols, axis = 1)

def featureDropper(self, cols):
    self.data = self.data.drop(cols, axis = 1)

def missingColCompare(self, train, validation):
    trainMissing = train.isnull().sum()
    trainMissing = trainMissing[trainMissing > 0].index

    validationMissing = validation.isnull().sum()
    validationMissing = validationMissing[validationMissing > 0].index

    print('Missing in validation, not train')
    print(set(validationMissing) - set(trainMissing))
    print('')
    print('Missing in train, not validation')
    print(set(trainMissing) - set(validationMissing))
