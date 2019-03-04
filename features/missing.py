
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