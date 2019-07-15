import numpy as np
import pandas as pd


def missingDataDropperAll(self):
    """
    Documentation:
        Description:
            Drops columns that include any missing data from self.data.
    """
    missingCols = self.data.isnull().sum()
    missingCols = missingCols[missingCols > 0].index
    self.data = self.data.drop(missingCols, axis=1)


def featureDropper(self, cols):
    """
    Documentation:
        Description:
            Drop a specific feature or set of features from self.data.
        Parameters:
            cols : list
                A list containing the column(s) to drop from self.data.
    """
    self.data = self.data.drop(cols, axis=1)


def missingColCompare(self, train, validation):
    """
    Documentation:
        Description:
            Compares the columns that contain missing data in the training dataset
            to the columns that contain missing data in the validation dataset. Prints 
            a summary of the disjunction.
        Parameters:
            train : Pandas DataFrame
                Training dataset.
            validation : Pandas DataFrame
                Validation dataset.
    """
    trainMissing = train.isnull().sum()
    trainMissing = trainMissing[trainMissing > 0].index

    validationMissing = validation.isnull().sum()
    validationMissing = validationMissing[validationMissing > 0].index

    print("Missing in validation, not train")
    print(set(validationMissing) - set(trainMissing))
    print("")
    print("Missing in train, not validation")
    print(set(trainMissing) - set(validationMissing))
