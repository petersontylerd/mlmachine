import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import sklearn.base as base

from collections import defaultdict

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


class Standard(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Apply standard scaling. Capable of fit_transforming new data, as well
            as transforming validation data with the same parameters.
        Parameters:
            cols : str, default = 'all'
                String describing which columns to scale. Takes values 'all',
                which will scale all columns and 'non-binary', which filters out
                dummy columns. These are identified by virtue of dummy columns
                having a data type of 'uint8'.
            train : boolean, default = True
                Controls whether to fit_transform training data or transform
                validation data using parameters learning from the fit on training
                data.
            trainValue : dict, default = None
                Dictionary containing 'feature : StandardScalar()' pairs to be used
                to transform validation data. Only used when train = False. Variable
                to be retrieved from traing pipeline is called trainValue_.
        Returns:
            X : array
                Dataset with standard scaled versions of input columns.
    """

    def __init__(self, cols="all", train=True, trainValue=None):
        self.cols = cols
        self.train = train
        self.trainValue = trainValue

        self.trainValue_ = defaultdict(preprocessing.StandardScaler)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.cols == "all":
            self.cols = X.columns
        elif self.cols == "non-binary":
            self.cols = X.select_dtypes(exclude="uint8").columns

        # Encode training data
        if self.train:

            # build dictionary of StandarScaler transformers
            self.trainValue_ = X[self.cols].apply(
                lambda x: self.trainValue_[x.name].fit(x.values.reshape(-1, 1))
            )

            # apply transformation to columns
            for col in self.cols:
                sc = self.trainValue_[col]
                X[col] = sc.transform(X[col].values.reshape(-1, 1))

        # apply transformation to columns
        else:
            for col in self.cols:
                sc = self.trainValue[col]
                X[col] = sc.transform(X[col].values.reshape(-1, 1))
        return X


class Robust(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Apply robust scaling. Capable of fit_transforming new data, as well as
            transforming validation data with the same parameters.
        Parameters:
            cols : str
                String describing which columns to scale. Takes values 'all', which
                will scale all columns and 'non-binary', which filters out dummy columns.
                These are identified by virtue of these columns having a data type of
                'uint8'.
            train : boolean, default = True
                Controls whether to fit_transform training data or transform validation
                data using parameters learned from the fit on training data.
            trainValue : dict, default = None
                Dictionary containing 'feature : RobustScalar()' pairs to be used
                to transform validation data. Only used when train = False.
                Variable to be retrieved from traing pipeline is called trainValue_.
        Returns:
            X : array
                Dataset with robust scaled versions of input columns.
    """

    def __init__(self, cols="all", train=True, trainValue=None):
        self.cols = cols
        self.train = train
        self.trainValue = trainValue

        self.trainValue_ = defaultdict(preprocessing.RobustScaler)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.cols == "all":
            self.cols = X.columns
        elif self.cols == "non-binary":
            self.cols = X.select_dtypes(exclude="uint8").columns

        # Encode training data
        if self.train:

            # build dictionary of StandarScaler transformers
            self.trainValue_ = X[self.cols].apply(
                lambda x: self.trainValue_[x.name].fit(x.values.reshape(-1, 1))
            )

            # apply transformation to columns
            for col in self.cols:
                sc = self.trainValue_[col]
                X[col] = sc.transform(X[col].values.reshape(-1, 1))

        # apply transformation to columns
        else:
            for col in self.trainValue.keys():
                sc = self.trainValue[col]
                X[col] = sc.transform(X[col].values.reshape(-1, 1))
        return X