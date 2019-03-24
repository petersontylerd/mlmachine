
import sklearn.base as base
import sklearn.preprocessing as preprocessing

import numpy as np
import pandas as pd

from collections import defaultdict

def cleanLabel(self, reverse = False):
    """
    Info:
        Description:
            a
        Parameters:
    """
    if self.targetType == 'continuous':
        self.y_ = self.y_.values.reshape(-1)
    elif self.targetType == 'categorical':
        self.le_ = prepocessing.LabelEncoder()

        self.y_ = self.le_.fit_transform(self.y_.values.reshape(-1))
        
        print('******************\nCategorical label encoding\n')
        for origLbl, encLbl in zip(np.sort(self.le_.classes_), np.sort(np.unique(self.y_))):
            print('{} --> {}'.format(origLbl, encLbl))

    if reverse:
        self.y_ = self.le_.inverse_transform(self.y_)

class Dummies(base.TransformerMixin, base.BaseEstimator):
    """
    Info:
        Description: 
            Create dummy columns for specified nominal 
            features.
    """
    def __init__(self, cols, dropFirst = True):
        self.cols = cols
        self.dropFirst = dropFirst
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = pd.get_dummies(data = X, columns = self.cols, drop_first = self.dropFirst)
        return X

class MissingDummies(base.TransformerMixin, base.BaseEstimator):
    """
    Info:
        Description: 
            If the test set is missing a level in a column, and
            therefore does now have the data necessary to create all dummy columns
            that were created in the test set, this class will add that dummy 
            column for the missing level and fill it with zeros.
        Parameters:

    """
    def __init__(self, trainCols):
        self.trainCols = trainCols
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        missingLevels = set(self.trainCols) - set(X.columns)
        for c in missingLevels:
            X[c] = 0
        X = X[self.trainCols]
        return X

class OrdinalEncoder(base.TransformerMixin, base.BaseEstimator):
    """
    Info:
        Description:
            Encode ordinal categorical columns. Capable of fit_transforming
            new data, as well as transforming validation data with the same
            encodings.
        Parameters:
            cols : list
                List of features to be encoded
            train : boolean, default = True
                Controls whether to fit_transform training data or
                transform validation data using encoder fit on 
                training data
            trainDict : dict, default = None
                Dictionary containing feature : LabelEncoder() pairs to be used
                to transform validation data. Only used when train = False.
                Variable to be retrieved is called colValueDict_
    """        
    def __init__(self, cols, train = True, trainDict = None):
        self.cols = cols
        self.train = train
        self.trainDict = trainDict

        self.colValueDict_ = defaultdict(preprocessing.LabelEncoder)
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # Encode training data
        if self.train:
            X[self.cols] = X[self.cols].apply(lambda x: self.colValueDict_[x.name].fit_transform(x))
        # Encode validation data with training data encodings.
        else:
            X[self.cols] = X[self.cols].apply(lambda x: self.trainDict[x.name].transform(x))
        return X

class CustomOrdinalEncoder(base.TransformerMixin, base.BaseEstimator):
    """
    Info:
        Description:
            Encode ordinal categorical columns using custom encodings. 
            
        Parameters:
            encodings : dict
                Dictionary of dictionaries containing columns / encoding
                instructions
    """        
    def __init__(self, encodings):
        self.encodings = encodings
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        for key, val in self.encodings.items():
            X[key] = X[key].replace(val)
        return X

