import sklearn.base as base
import sklearn.preprocessing as preprocessing

import numpy as np
import pandas as pd

from collections import defaultdict


def cleanLabel(self, reverse=False):
    """
    Documentation:
        Description:
            Encode label into numerical form.
        Parameters:
            reverse : boolean, default = False
                Reverses encoding of target variables back to original variables.
    """
    if self.targetType == "categorical":
        self.le_ = preprocessing.LabelEncoder()

        self.target = pd.Series(
            self.le_.fit_transform(self.target.values.reshape(-1)),
            name=self.target.name,
            index=self.target.index
        )

        print("******************\nCategorical label encoding\n")
        for origLbl, encLbl in zip(
            np.sort(self.le_.classes_), np.sort(np.unique(self.target))
        ):
            print("{} --> {}".format(origLbl, encLbl))

    if reverse:
        self.target = self.le_.inverse_transform(self.target)


class Dummies(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Create dummy columns for specified nominal features.
        Parameters:
            cols : list
                List containing column(s) to be transformed.
            dropFirst : boolean, default = True
                Dictates whether pd.get_dummies automatically drops of the dummy columns.
                Help to avoid multicollinearity.
            train : boolean, default = True
                Controls whether to fit_transform training data or transform
                validation data using encoder fit on training data.
        Attributes:
            originalCols : Pandas DataFrame
                Original columns prior to
        Returns:
            X : array
                Dataset with dummy column representation of input variables.
    """

    def __init__(self, cols, dropFirst=True, train=True):
        self.cols = cols
        self.dropFirst = dropFirst
        self.train = train

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.train:
            self.originalCols_ = X[self.cols]

        # transform data into dolumn column representation
        X = pd.get_dummies(data=X, columns=self.cols, drop_first=self.dropFirst)
        return X


class OrdinalEncoder(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Encode ordinal categorical columns. Capable of fit_transforming
            new data, as well as transforming validation data with the same
            encodings used on the training data.
        Parameters:
            cols : list
                List of features to be encoded.
            train : boolean, default = True
                Controls whether to fit_transform training data or transform
                validation data using encoder fit on training data.
            trainValue : dict, default = None
                Dictionary containing 'feature : LabelEncoder()' pairs to be used
                to transform validation data. Only used when train = False.
                Variable to be retrieved from traing pipeline is called trainValue_.
        Returns:
            X : array
                Dataset with encoded versions of input variables.

    """

    def __init__(self, cols, train=True, trainValue=None):
        self.cols = cols
        self.train = train
        self.trainValue = trainValue

        self.trainValue_ = defaultdict(preprocessing.LabelEncoder)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Encode training data
        if self.train:
            X[self.cols] = X[self.cols].apply(
                lambda x: self.trainValue_[x.name].fit_transform(x)
            )
        # Encode validation data with training data encodings.
        else:
            X[self.cols] = X[self.cols].apply(
                lambda x: self.trainValue[x.name].transform(x)
            )
        return X


class CustomOrdinalEncoder(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Encode ordinal categorical columns using custom encodings. This is useful
            for ensuring an ordinal column is encoded in a specific order.
        Parameters:
            encodings : dict
                Dictionary of dictionaries containing 'columns : encoding'
                pairs that proscribe encoding instructions.
        Returns:
            X : array
                Dataset with encoded versions of input variables.
    """

    def __init__(self, encodings):
        self.encodings = encodings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for key, val in self.encodings.items():
            X[key] = X[key].replace(val)
        return X

