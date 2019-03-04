
import numpy as np
import pandas as pd
import sklearn.preprocessing as prepocessing
import sklearn.base as base


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
