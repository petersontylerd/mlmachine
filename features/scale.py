
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import sklearn.base as base

from collections import defaultdict

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action = 'ignore', category = DataConversionWarning)

class Standard(base.TransformerMixin, base.BaseEstimator):
    """
    Info:
        Description:
            Apply standard scaling. Capable of fit_transforming new data, 
            as well as transforming validation data with the same
            parameters.
        Parameters:
            cols : str
                String describing which columns to scale. Takes values 'all',
                which will scale all columns and 'non-binary', which filters out
                dummy columns. These are identified by virtue of these columns
                having a data type of 'uint8'.
            train : boolean, default = True
                Controls whether to fit_transform training data or
                transform validation data using fit on training data
            trainDict : dict, default = None
                Dictionary containing feature : StandardScalar() pairs to be used
                to transform validation data. Only used when train = False.
                Variable to be retrieved is called colValueDict_
    """        
    def __init__(self, cols = 'all', train = True, trainDict = None):
        self.cols = cols
        self.train = train
        self.trainDict = trainDict

        self.colValueDict_ = defaultdict(preprocessing.StandardScaler)

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        if self.cols == 'all':
            self.cols = X.columns
        elif self.cols == 'non-binary':
            self.cols = X.select_dtypes(exclude = 'uint8').columns
        
        # Encode training data
        if self.train:
            
            # build dictionary of StandarScaler transformers
            self.colValueDict_ = X[self.cols].apply(lambda x: self.colValueDict_[x.name].fit(x.values.reshape(-1, 1)))
            
            # apply transformation to columns
            for col in self.cols:
                sc = self.colValueDict_[col]
                X[col] = sc.transform(X[col].values.reshape(-1, 1))
            
        # apply transformation to columns
        else:
            for col in self.cols:
                sc = self.trainDict[col]
                X[col] = sc.transform(X[col].values.reshape(-1, 1))
        return X

class Robust(base.TransformerMixin, base.BaseEstimator):
    """
    Info:
        Description:
            Apply robust scaling. Capable of fit_transforming new data, 
            as well as transforming validation data with the same
            parameters.
        Parameters:
            cols : str
                String describing which columns to scale. Takes values 'all',
                which will scale all columns and 'non-binary', which filters out
                dummy columns. These are identified by virtue of these columns
                having a data type of 'uint8'.
            train : boolean, default = True
                Controls whether to fit_transform training data or
                transform validation data using fit on training data
            trainDict : dict, default = None
                Dictionary containing feature : RobustScalar() pairs to be used
                to transform validation data. Only used when train = False.
                Variable to be retrieved is called colValueDict_
    """        
    def __init__(self, cols = 'all', train = True, trainDict = None):
        self.cols = cols
        self.train = train
        self.trainDict = trainDict

        self.colValueDict_ = defaultdict(preprocessing.RobustScaler)

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        if self.cols == 'all':
            self.cols = X.columns
        elif self.cols == 'non-binary':
            self.cols = X.select_dtypes(exclude = 'uint8').columns
        
        # Encode training data
        if self.train:
            
            # build dictionary of StandarScaler transformers
            self.colValueDict_ = X[self.cols].apply(lambda x: self.colValueDict_[x.name].fit(x.values.reshape(-1, 1)))
            
            # apply transformation to columns
            for col in self.cols:
                sc = self.colValueDict_[col]
                X[col] = sc.transform(X[col].values.reshape(-1, 1))
            
        # apply transformation to columns
        else:
            for col in self.cols:
                sc = self.trainDict[col]
                X[col] = sc.transform(X[col].values.reshape(-1, 1))
        return X

