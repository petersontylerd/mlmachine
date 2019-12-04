import numpy as np
import pandas as pd

from scipy import stats
from scipy import special

import sklearn.base as base
import sklearn.preprocessing as preprocessing


class equal_width_binner(base.TransformerMixin, base.BaseEstimator):
    """
    documentation:
        description:
            bin numeric columns into specified segments. bins training data
            features, and stores the cut points to be used on validation and
            unseen data.
        parameters:
            equal_bin_dict : dictionary, default =None
                dictionary containing 'column : label' pairs. label is a list that
                proscribes the bin labels to be used for each paired column. the bin
                size is calculated based off of the the number of labels. the labels
                are expected to be a list that describes how the bins should be named,
                i.e. a label list of ['low','med','high'] will instruct the binner to
                create three bins and then call each bin 'low','med' and 'high'.
            train : boolean, default=True
                tells class whether we are binning training data or unseen data.
            train_value : dict, default =None
                dictionary containing 'feature : mode' pairs to be used to transform
                validation data. only used when train=False. retrieved from training
                data pipeline using named steps. variable to be retrieved from traing
                pipeline is called train_value_..
        returns:
            X : array
                dataset with additional columns represented binned versions of input columns.
    """

    def __init__(self, equal_bin_dict=None, train=True, train_value=None):
        self.equal_bin_dict = equal_bin_dict
        self.train = train
        self.train_value = train_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode training data
        if self.train:

            # create shell dictionary to store learned bins for each column
            self.train_value_ = {}

            # iterate through column : label pairs
            for col, label in self.equal_bin_dict.items():

                # retrieve bin cutoffs from original column
                _, bins = pd.cut(X[col], bins=len(label), labels=label, retbins=True)

                # add binned version of original column to dataset
                X["{}{}".format(col, "equal_bin")] = pd.cut(
                    X[col], bins=len(label), labels=label
                )

                # build col_value_dict
                self.train_value_[col] = bins

        # for each column, bin the values based on the cut_offs learned on training data
        else:
            # todo - does not currently apply bin label. just the interval index range. not usable.
            # iterate through columns and stored bins that were learned from training data
            for col, bins in self.train_value.items():
                train_bins = pd.interval_index.from_breaks(bins)
                print(bins)
                print(train_bins)
                print(type(train_bins))
                X["{}{}".format(col, "equal_bin")] = pd.cut(X[col], bins=train_bins)
        return X


class percentile_binner(base.TransformerMixin, base.BaseEstimator):
    """
    documentation:
        description:
            bin numeric columns into segments based on percentile cut_offs.
        parameters:
            columns : list
                list of colummns to be binned. the percentiles are derived from
                the raw data.
            percs : list
                percentiles for determining cut_off points for bins.
            train : boolean, default=True
                tells class whether we are binning training data or unseen data.
            train_value : dict, default =None
                dictionary containing 'feature : mode' pairs to be used to transform
                validation data. only used when train=False. retrieved from training
                data pipeline using named steps. variable to be retrieved from traing
                pipeline is called train_value_..
        returns:
            X : array
                dataset with additional columns represented binned versions of input columns.
    """

    def __init__(self, columns=None, percs=None, train=True, train_value=None):
        self.columns = columns
        self.percs = percs
        self.train = train
        self.train_value = train_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # bin training data
        if self.train:

            # create shell dictionary to store percentile values for each column
            self.train_value_ = {}

            # iterate through columns by name
            for col in self.columns:
                # create empty perc_bin column
                bin_col = "{}perc_bin".format(col)
                X[bin_col] = np.nan

                # determine percentile cut_offs
                perc_vals = []
                for perc in self.percs:
                    perc_vals.append(np.percentile(X[col], perc))

                # iterate through custom binning
                for ix, ceil in enumerate(perc_vals):
                    # first item
                    if ix == 0:
                        X.loc[X[col] <= ceil, bin_col] = ix

                    # next to last and last item
                    elif ix == len(perc_vals) - 1:
                        X.loc[(X[col] > floor) & (X[col] <= ceil), bin_col] = ix
                        X.loc[X[col] > ceil, bin_col] = ix + 1
                    # everything in between
                    else:
                        X.loc[(X[col] > floor) & (X[col] <= ceil), bin_col] = ix

                    # increment the floor
                    floor = ceil

                # build col_value_dict
                self.train_value_[col] = perc_vals

                # set data type
                X[bin_col] = X[bin_col].astype("int64")

        # bin validation data based on percentile values learned from training data
        else:
            # iterate through columns by name
            for col in self.train_value.keys():
                # create empty perc_bin column
                bin_col = "{}perc_bin".format(col)
                X[bin_col] = np.nan

                # iterate through bin values
                for ix, ceil in enumerate(self.train_value[col]):
                    # first item
                    if ix == 0:
                        X.loc[X[col] <= ceil, bin_col] = ix
                    # next to last and last item
                    elif ix == len(self.train_value[col]) - 1:
                        X.loc[(X[col] > floor) & (X[col] <= ceil), bin_col] = ix
                        X.loc[X[col] > ceil, bin_col] = ix + 1
                    # everything in between
                    else:
                        X.loc[(X[col] > floor) & (X[col] <= ceil), bin_col] = ix

                    # increment the floor
                    floor = ceil

                # set data type
                X[bin_col] = X[bin_col].astype("int64")
        return X


class custom_binner(base.TransformerMixin, base.BaseEstimator):
    """
    documentation:
        description:
            bin numeric columns into custom segments.
        parameters:
            custom_bin_dict : dictionary
                dictionary containing 'column : bin' specifcation pairs. bin specifications
                should be a list.
        returns:
            X : array
                dataset with additional columns represented binned versions of input columns.
    """

    def __init__(self, custom_bin_dict):
        self.custom_bin_dict = custom_bin_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # iterate through columns by name
        for col in self.custom_bin_dict.keys():
            # create empty custom_bin column
            bin_col = "{}custom_bin".format(col)
            X[bin_col] = np.nan

            # append feature_dtype dict
            # self.feature_type['categorical'].append(bin_col)

            # iterate through custom binning
            for ix, ceil in enumerate(self.custom_bin_dict[col]):
                # first item
                if ix == 0:
                    X.loc[X[col] <= ceil, bin_col] = ix
                # next to last and last item
                elif ix == len(self.custom_bin_dict[col]) - 1:
                    X.loc[(X[col] > floor) & (X[col] <= ceil), bin_col] = ix
                    X.loc[X[col] > ceil, bin_col] = ix + 1
                # everything in between
                else:
                    X.loc[(X[col] > floor) & (X[col] <= ceil), bin_col] = ix

                # increment the floor
                floor = ceil

            # set data type
            X[bin_col] = X[bin_col].astype("int64")
        return X
