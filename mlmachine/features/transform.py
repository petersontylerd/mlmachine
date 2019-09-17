import numpy as np
import pandas as pd

from scipy import stats
from scipy import special

import sklearn.base as base
import sklearn.preprocessing as preprocessing


class EqualWidthBinner(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Bin continuous columns into specified segments. Bins training data
            features, and stores the cut points to be used on validation and
            unseen data.
        Parameters:
            equalBinDict : dictionary, default = None
                Dictionary containing 'column : label' pairs. Label is a list that
                proscribes the bin labels to be used for each paired column. The bin
                size is calculated based off of the the number of labels. The labels
                are expected to be a list that describes how the bins should be named,
                i.e. a label list of ['low','med','high'] will instruct the binner to
                create three bins and then call each bin 'low','med' and 'high'.
            train : boolean, default = True
                Tells class whether we are binning training data or unseen data.
            trainValue : dict, default = None
                Dictionary containing 'feature : mode' pairs to be used to transform
                validation data. Only used when train = False. Retrieved from training
                data pipeline using named steps. Variable to be retrieved from traing
                pipeline is called trainValue_..
        Returns:
            X : array
                Dataset with additional columns represented binned versions of input columns.
    """

    def __init__(self, equalBinDict=None, train=True, trainValue=None):
        self.equalBinDict = equalBinDict
        self.train = train
        self.trainValue = trainValue

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Encode training data
        if self.train:

            # create shell dictionary to store learned bins for each column
            self.trainValue_ = {}

            # iterate through column : label pairs
            for col, label in self.equalBinDict.items():

                # retrieve bin cutoffs from original column
                _, bins = pd.cut(X[col], bins=len(label), labels=label, retbins=True)

                # add binned version of original column to dataset
                X["{}{}".format(col, "EqualBin")] = pd.cut(
                    X[col], bins=len(label), labels=label
                )

                # build colValueDict
                self.trainValue_[col] = bins

        # For each column, bin the values based on the cut-offs learned on training data
        else:
            # TODO - does not currently apply bin label. just the interval index range. not usable.
            # iterate through columns and stored bins that were learned from training data
            for col, bins in self.trainValue.items():
                trainBins = pd.IntervalIndex.from_breaks(bins)
                print(bins)
                print(trainBins)
                print(type(trainBins))
                X["{}{}".format(col, "EqualBin")] = pd.cut(X[col], bins=trainBins)
        return X


class PercentileBinner(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Bin continuous columns into segments based on percentile cut-offs.
        Parameters:
            cols : list
                List of colummns to be binned. The percentiles are derived from
                the raw data.
            percs : list
                Percentiles for determining cut-off points for bins.
            train : boolean, default = True
                Tells class whether we are binning training data or unseen data.
            trainValue : dict, default = None
                Dictionary containing 'feature : mode' pairs to be used to transform
                validation data. Only used when train = False. Retrieved from training
                data pipeline using named steps. Variable to be retrieved from traing
                pipeline is called trainValue_..
        Returns:
            X : array
                Dataset with additional columns represented binned versions of input columns.
    """

    def __init__(self, cols=None, percs=None, train=True, trainValue=None):
        self.cols = cols
        self.percs = percs
        self.train = train
        self.trainValue = trainValue

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # bin training data
        if self.train:

            # create shell dictionary to store percentile values for each column
            self.trainValue_ = {}

            # iterate through columns by name
            for col in self.cols:
                # create empty PercBin column
                binCol = "{}PercBin".format(col)
                X[binCol] = np.nan

                # determine percentile cut-offs
                percVals = []
                for perc in self.percs:
                    percVals.append(np.percentile(X[col], perc))

                # iterate through custom binning
                for ix, ceil in enumerate(percVals):
                    # first item
                    if ix == 0:
                        X.loc[X[col] <= ceil, binCol] = ix

                    # next to last and last item
                    elif ix == len(percVals) - 1:
                        X.loc[(X[col] > floor) & (X[col] <= ceil), binCol] = ix
                        X.loc[X[col] > ceil, binCol] = ix + 1
                    # everything in between
                    else:
                        X.loc[(X[col] > floor) & (X[col] <= ceil), binCol] = ix

                    # increment the floor
                    floor = ceil

                # build colValueDict
                self.trainValue_[col] = percVals

                # set data type
                X[binCol] = X[binCol].astype("int64")

        # bin validation data based on percentile values learned from training data
        else:
            # iterate through columns by name
            for col in self.trainValue.keys():
                # create empty PercBin column
                binCol = "{}PercBin".format(col)
                X[binCol] = np.nan

                # iterate through bin values
                for ix, ceil in enumerate(self.trainValue[col]):
                    # first item
                    if ix == 0:
                        X.loc[X[col] <= ceil, binCol] = ix
                    # next to last and last item
                    elif ix == len(self.trainValue[col]) - 1:
                        X.loc[(X[col] > floor) & (X[col] <= ceil), binCol] = ix
                        X.loc[X[col] > ceil, binCol] = ix + 1
                    # everything in between
                    else:
                        X.loc[(X[col] > floor) & (X[col] <= ceil), binCol] = ix

                    # increment the floor
                    floor = ceil

                # set data type
                X[binCol] = X[binCol].astype("int64")
        return X


class CustomBinner(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Bin continuous columns into custom segments.
        Parameters:
            customBinDict : dictionary
                Dictionary containing 'column : bin' specifcation pairs. Bin specifications
                should be a list.
        Returns:
            X : array
                Dataset with additional columns represented binned versions of input columns.
    """

    def __init__(self, customBinDict):
        self.customBinDict = customBinDict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # iterate through columns by name
        for col in self.customBinDict.keys():
            # create empty CustomBin column
            binCol = "{}CustomBin".format(col)
            X[binCol] = np.nan

            # append featureDtype dict
            # self.featureByDtype['categorical'].append(binCol)

            # iterate through custom binning
            for ix, ceil in enumerate(self.customBinDict[col]):
                # first item
                if ix == 0:
                    X.loc[X[col] <= ceil, binCol] = ix
                # next to last and last item
                elif ix == len(self.customBinDict[col]) - 1:
                    X.loc[(X[col] > floor) & (X[col] <= ceil), binCol] = ix
                    X.loc[X[col] > ceil, binCol] = ix + 1
                # everything in between
                else:
                    X.loc[(X[col] > floor) & (X[col] <= ceil), binCol] = ix

                # increment the floor
                floor = ceil

            # set data type
            X[binCol] = X[binCol].astype("int64")
        return X


def featureDropper(self, cols, data, featureByDtype):
    """
    Documentation:
        Description:
            Removes feature from dataset and from self.featureByDtype.
        Parameters:
            cols : list
                List of features to be dropped.
            data : Pandas DataFrame, default = None
                Pandas DataFrame containing independent variables.
            featureByDtype : dictionary, default = None
                Dictionary containing string/list key/value pairs, where the key is the feature
                type and the value is a list of features of that type.
        Returns:
            data : Pandas DataFrame
                Modified input with all specified columns data removed.
            featureByDtype : dictionary
                Modified input with all specified column names data removed.
    """
    for col in cols:
        # delete colummn from data from
        data = data.drop([col], axis=1)

        # delete column name from featureByDtype dict
        if col in featureByDtype["categorical"]:
            featureByDtype["categorical"].remove(col)
        elif col in featureByDtype["continuous"]:
            featureByDtype["continuous"].remove(col)

    return data, featureByDtype

class NumericCoercer(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Transform all columns with non-numeric data types to numeric.
        Returns:
            X : array
                Dataset with encoded versions of input variables.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = X.select_dtypes(exclude=['number']).columns
        for col in cols:
            X[col] = X[col].apply(pd.to_numeric)
        return X