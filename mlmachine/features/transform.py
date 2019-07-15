import numpy as np
import pandas as pd

from scipy import stats
from scipy import special

import sklearn.base as base


def skewSummary(self):
    """
    Documentation:
        Description:
            Displays Pandas DataFrame summarizing the skew of each continuous variable. Also summarizes
            the percent of a column that has a value of zero.
    """
    skewness = (
        self.data[self.featureByDtype_["continuous"]]
        .apply(lambda x: stats.skew(x.dropna()))
        .sort_values(ascending=False)
    )
    skewness = pd.DataFrame({"Skew": skewness})

    # add column describing percent of values that are zero
    skewness["PctZero"] = np.nan
    for col in self.featureByDtype_["continuous"]:

        try:
            skewness.loc[col]["PctZero"] = self.data[self.data[col] == 0][
                col
            ].value_counts() / len(self.data)
        except ValueError:
            skewness.loc[col]["PctZero"] = 0.0
    skewness = skewness.sort_values(["Skew"])
    display(skewness)


class SkewTransform(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Performs Box-Cox + 1 transformation on continuous features with an absolute skew 
            value above skewMin. The lambda chosen for each feature is the value that gets the 
            skew closest to zero. The same lambda values are reused on unseen data.
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
                using Box-Cox + 1 transformation. Only used when train = False. Variable to be retrieved 
                from train pipeline from traing pipeline is called colValueDict_..
        Returns:
            X : array
                Box-Cox + 1 transformed input data.
    """

    def __init__(
        self, cols=None, skewMin=None, pctZeroMax=None, train=True, trainDict=None
    ):
        self.cols = cols
        self.skewMin = skewMin
        self.pctZeroMax = pctZeroMax
        self.train = train
        self.trainDict = trainDict

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
            self.colValueDict_ = {}
            for col, skew in skews.items():
                # determine percent of column values that are zero
                try:
                    pctZero = (X[X[col] == 0][col].value_counts() / len(X)).values[0]

                except IndexError:
                    pctZero = 0.0

                # if the skew value is greater than the minimum skew allowed and pctZero is lower than a maximum allowed
                if skew >= self.skewMin and pctZero <= self.pctZeroMax:

                    # build dictionary of lambda : skew pairs
                    lambdaDict = {}
                    for lmbda in np.linspace(-2.0, 2.0, 500):
                        lambdaDict[lmbda] = abs(
                            stats.skew(special.boxcox1p(X[col], lmbda))
                        )

                    # detemine value of lambda that result in a skew closest to 0

                    lowLambda = min(
                        lambdaDict.items(), key=lambda kv: abs(kv[1] - 0.0)
                    )[0]

                    X[col] = special.boxcox1p(X[col], lowLambda)
                    self.colValueDict_[col] = lowLambda
                    # print('transformed {}'.format(col))

        # Encode validation data with training data encodings.
        else:
            for col, lmbda in self.trainDict.items():
                X[col] = special.boxcox1p(X[col], lmbda)
        return X


class EqualBinner(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Bin continuous columns into specified segments. Bins training data 
            features, and stores the cut points to be used on validation and
            unseen data.
        Parameters:
            equalBinDict : dictionary, default = None
                Dictionary containing 'column : label' pairs. Custom bin labels to be 
                used for each paired column. The bin size is calculated based off 
                of the label length. The labels are expected to be a list that 
                describes how the bins should be named, i.e. a label list of
                ['low','med','high'] will instruct the binner to create three bins
                and then call each bin 'low','med' and 'high'.
            train : boolean, default = True
                Tells class whether we are binning training data or unseen data.
            trainDict : dict, default = None
                Dictionary containing 'feature : mode' pairs to be used to transform 
                validation data. Only used when train = False. Retrieved from training 
                data pipeline using named steps. Variable to be retrieved from traing 
                pipeline is called colValueDict_..
        Returns:
            X : array
                Dataset with additional columns represented binned versions of input columns.
    """

    def __init__(self, equalBinDict=None, train=True, trainDict=None):
        self.equalBinDict = equalBinDict
        self.train = train
        self.trainDict = trainDict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Encode training data
        if self.train:

            # create shell dictionary to store learned bins for each column
            self.trainDict_ = {}

            # iterate through column : label pairs
            for col, label in self.equalBinDict.items():

                # retrieve bin cutoffs from original column
                _, bins = pd.cut(X[col], bins=len(label), labels=label, retbins=True)

                # add binned version of original column to dataset
                X["{}{}".format(col, "EqualBin")] = pd.cut(
                    X[col], bins=len(label), labels=label
                )

                # append featureDtype dict
                # self.featureByDtype_['categorical'].append('{}{}'.format(col,'EqualBin'))

                # build colValueDict
                self.trainDict_[col] = bins

                # set data type
                # X['{}{}'.format(col,'EqualBin')] = X['{}{}'.format(col,'EqualBin')].astype('int64')

        # For each column, bin the values based on the cut-offs learned on training data
        else:
            # iterate through columns and stored bins that were learned from training data
            for col, bins in self.trainDict.items():
                trainBins = pd.IntervalIndex.from_breaks(bins)
                X["{}{}".format(col, "EqualBin")] = pd.cut(X[col], bins=trainBins)

                # append featureDtype dict
                # self.featureByDtype_['categorical'].append('{}{}'.format(col,'EqualBin'))

                # set data type
                # X['{}{}'.format(col,'EqualBin')] = X['{}{}'.format(col,'EqualBin')].astype('int64')
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
            trainDict : dict, default = None
                Dictionary containing 'feature : mode' pairs to be used to transform 
                validation data. Only used when train = False. Retrieved from training 
                data pipeline using named steps. Variable to be retrieved from traing 
                pipeline is called colValueDict_..
        Returns:
            X : array
                Dataset with additional columns represented binned versions of input columns.
    """

    def __init__(self, cols=None, percs=None, train=True, trainDict=None):
        self.cols = cols
        self.percs = percs
        self.train = train
        self.trainDict = trainDict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # bin training data
        if self.train:

            # create shell dictionary to store percentile values for each column
            self.trainDict_ = {}

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
                self.trainDict_[col] = percVals

                # set data type
                X[binCol] = X[binCol].astype("int64")

        # bin validation data based on percentile values learned from training data
        else:
            # iterate through columns by name
            for col in self.trainDict.keys():
                # create empty PercBin column
                binCol = "{}PercBin".format(col)
                X[binCol] = np.nan

                # iterate through bin values
                for ix, ceil in enumerate(self.trainDict[col]):
                    # first item
                    if ix == 0:
                        X.loc[X[col] <= ceil, binCol] = ix
                    # next to last and last item
                    elif ix == len(self.trainDict[col]) - 1:
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
            # self.featureByDtype_['categorical'].append(binCol)

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


def featureDropper(self, cols):
    """
    Documentation:
        Description:
            Removes feature from dataset and from self.featureByDtype_.
        Parameters:
            cols : list
                List of features to be dropped.
    """
    for col in cols:
        # delete colummn from data from
        self.data = self.data.drop([col], axis=1)

        # delete column name from featureByDtype dict
        if col in self.featureByDtype_["categorical"]:
            self.featureByDtype_["categorical"].remove(col)
        elif col in self.featureByDtype_["continuous"]:
            self.featureByDtype_["continuous"].remove(col)

