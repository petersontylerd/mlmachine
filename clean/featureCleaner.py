
import numpy as np
import pandas as pd
import sklearn.preprocessing as prepocessing
import sklearn.base as base

def transformLabel(self, reverse = False):
        """

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

def missingDataSummary(self):
    """

    """
    self.missingCols_ = [col for col in self.X_.columns if self.X_[col].isnull().any()]
    
    print(self.X_[self.missingCols_].isnull().sum())
    

def featureDropper(self, cols):
    self.X_ = self.X_.drop(cols, axis = 1)

def missingDataDropperAll(self):
    self.missingDataSummary()
    self.X_ = self.X_.drop(self.missingCols_, axis = 1)


class DataFrameImputer(base.TransformerMixin, base.BaseEstimator):

    def __init__(self):
        """
        Info:
            Description: Impute missing values. Columns of dtype object are imputed with
            the most frequent value in column. Columns of other types are imputed with 
            mean of column.
        """
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X]
        ,index = X.columns)
        return X.fillna(fill)
