import numpy as np
import pandas as pd

from scipy import stats
from scipy import special

import sklearn.base as base


def skewSummary(self, data=None, columns=None):
    """
    Documentation:
        Description:
            Displays Pandas DataFrame summarizing the skew of each continuous variable. Also summarizes
            the percent of a column that has a value of zero.
        Parameters:
            data : Pandas DataFrame, default = None
                Pandas DataFrame containing independent variables. If left as None,
                the feature dataset provided to Machine during instantiation is used.
            columns : list of strings, default = None
                List containing string names of columns. If left as None, the value associated
                with sel.featureByDtype["continuous"] will be used as the column list.
    """
    # use data/featureByDtype["continuous"] columns provided during instantiation if left unspecified
    if data is None:
        data = self.data
    if columns is None:
        columns = self.featureByDtype["continuous"]

    skewness = (
        data[columns]
        .apply(lambda x: stats.skew(x.dropna()))
        .sort_values(ascending=False)
    )
    skewness = pd.DataFrame({"Skew": skewness})

    # add column describing percent of values that are zero
    skewness["PctZero"] = np.nan
    for col in columns:

        try:
            skewness.loc[col]["PctZero"] = data[data[col] == 0][
                col
            ].value_counts() / len(data)
        except ValueError:
            skewness.loc[col]["PctZero"] = 0.0
    skewness = skewness.sort_values(["Skew"])
    display(skewness)


class SkewTransform(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Performs Box-Cox (or Box-Cox + 1) transformation on continuous features with a skew
            value above skewMin. The lambda chosen for each feature is the value that maximizes
            the log-likelihood function. The same lambda values are reused on unseen data.
        Parameters:
            cols : list, default = None
                List of columns to be evaluated for transformation.
            skewMin : float, default = None
                Minimum absolute skew of feature needed in order for feature to be transformed.
            pctZeroMax : float, default = None
                Maximum percent zero values a column is allowed to have in order to be transformed.
                Must be a value between 0.0 and 1.0.
            train : boolean, default = True
                Controls whether to fit_transform training data or transform validation data using
                lambdas determined from training data.
            classDict : dict, default = None
                Dictionary containing 'feature : lambda' pairs to be used to transform validation data
                using Box-Cox (or Box-Cox + 1) transformation. Only used when train = False. Variable
                to be retrieved from train pipeline from traing pipeline is called trainValue_.
            verbose : boolean, default = False
                If True display previous and post-transformation skew values for each feature.
        Returns:
            X : array
                Box-Cox (or Box-Cox + 1) transformed input data.
    """

    def __init__(
        self, cols=None, skewMin=None, pctZeroMax=None, train=True, trainValue=None, verbose = False
    ):
        self.cols = cols
        self.skewMin = skewMin
        self.pctZeroMax = pctZeroMax
        self.train = train
        self.trainValue = trainValue
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Encode training data
        if self.train:
            skews = (
                X[self.cols]
                .apply(lambda x: stats.skew(x.dropna()))
                .sort_values(ascending=False)
                .to_dict()
            )
            self.trainValue_ = {}
            for col, skew in skews.items():
                # determine percent of column values that are zero
                try:
                    pctZero = (X[X[col] == 0][col].value_counts() / len(X)).values[0]

                except IndexError:
                    pctZero = 0.0

                # if the skew value is greater than the minimum skew allowed and pctZero is lower than a maximum allowed
                if skew >= self.skewMin and pctZero <= self.pctZeroMax:

                    try:
                        X[col], lmbda = stats.boxcox(X[col], lmbda = None)
                    except ValueError:
                        X[col], lmbda = stats.boxcox(X[col] + 1, lmbda = None)

                    self.trainValue_[col] = lmbda

                    if self.verbose:
                        print('{}: {:.5f} --> {:.5f}'.format(col,skew,stats.skew(X[col].dropna())))

        # transform data with lambda learned on training data
        else:
            for col, lmbda in self.trainValue.items():
                try:
                    X[col] = stats.boxcox(X[col], lmbda = lmbda)
                except ValueError:
                    X[col] = stats.boxcox(X[col] + 1, lmbda = lmbda)
        return X


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