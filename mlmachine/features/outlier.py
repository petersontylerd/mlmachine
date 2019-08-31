import numpy as np
import pandas as pd

import sklearn.base as base

from collections import Counter

import eif

class OutlierIQR(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Identifies outliers using inter-quartile range method.
        Parameters:
            outlierCount : int
                Minimum number of values across all features that need to be outliers
                in order for an observation to be flagged.
            iqrStep : float
                Multiplier that controls level of sensitivity of outlier detection method.
                Higher values for iqrStep will cause OutlierIQR to only detect increasingly
                extreme values.
            features : list
                List of features to be evaluated for outliers.
            dropOutliers : boolean, default = False
                If True, drops outliers from the input data.
        Returns:
            X : array
                Dataset with outlier observations removed.
    """

    def __init__(self, outlierCount, iqrStep, features, dropOutliers=False):
        self.outlierCount = outlierCount
        self.iqrStep = iqrStep
        self.features = features
        self.dropOutliers = dropOutliers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        outlier_indices = []

        # iterate over features(columns)
        for col in self.features:
            Q1 = np.percentile(X[col], 25)
            Q3 = np.percentile(X[col], 75)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = self.iqrStep * IQR

            # Determine a list of indices of outliers for feature col
            outlier_list_col = X[
                (X[col] < Q1 - outlier_step) | (X[col] > Q3 + outlier_step)
            ].index

            # append the found outlier indices for col to the list of outlier indices
            outlier_indices.extend(outlier_list_col)

        # select observations containing more than 'outlierCount' outliers
        outlier_indices = Counter(outlier_indices)

        self.outliers_ = list(
            k for k, v in outlier_indices.items() if v >= self.outlierCount
        )

        if self.dropOutliers:
            X = X.drop(self.outliers_, axis=0).reset_index(drop=True)

        return X


class ExtendedIsoForest(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Identifies outliers using Extended Isolation Forest method.
        Parameters:
            cols : list
                Columns to be evaluated by Extended Isolation Forest
            nTrees : int
                Number of trees to be used.
            sampleSize : int
                Sub-sample size for creating each trees. Values must be smaller that input dataset
                row count.
            ExtensionLevel : int
                Degrees of freedom for choosing hyperplanes that divide data. Value must be smaller
                than input dataset column count.
            anomaliesRatio : float
                Percent of input dataset observations to identify as outliers.
            dropOutliers : boolean, default = False
                Dictates whether identified outliers are removed from input dataset.
        Returns:
            X : array
                Dataset with outlier observations removed.
    """

    def __init__(self, cols, nTrees, sampleSize, ExtensionLevel, anomaliesRatio, dropOutliers=False):
        self.cols= cols
        self.nTrees = nTrees
        self.sampleSize = sampleSize
        self.ExtensionLevel = ExtensionLevel
        self.anomaliesRatio = anomaliesRatio
        self.dropOutliers = dropOutliers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        extIso = eif.iForest(
            X = X[self.cols].values,
            ntrees=self.nTrees,
            sample_size=self.sampleSize,
            ExtensionLevel=self.ExtensionLevel,
        )

        # calculate anomaly scores
        anomalyScores = extIso.compute_paths(
            X_in=X[self.cols].values
        )

        anomalyScoresSorted = pd.DataFrame(
            anomalyScores,
            index = X.index,
            columns = ['Anomaly score']
        ).sort_values(
            ['Anomaly score'],
            ascending = False
        )

        self.outliers_ = np.array(
            anomalyScoresSorted[:int(np.ceil(self.anomaliesRatio * X.shape[0]))].index
        )

        if self.dropOutliers:
            X = X.drop(self.outliers_, axis=0).reset_index(drop=True)

        return X


def outlierSummary(self, iqrOutliers, ifOutliers, eifOutliers):
    """
    Documentation:
        Description:
            Creates Pandas DataFrame summarizing which observations were flagged
            as outliers and by which outlier detection method each observation was
            identified.
        Parameters:
            iqrOutliers : array
                Array contains indexes of observations identified as outliers using
                IQR method.
            ifOutliers : array
                Array contains indexes of observations identified as outliers using
                Isolition Forest method.
            eifOutliers : array
                Array contains indexes of observations identified as outliers using
                Extended Isolition Forest method.
        Returns:
            outlierSummary : Pandas DataFrame
                DataFrame summarizing outlier
    """
    # merge and de-duplicate outlier index values
    outlierIxs = np.unique(np.concatenate([iqrOutliers, ifOutliers, eifOutliers]))

    # create shell dataframe
    outlierSummary = pd.DataFrame(
        columns = ['IQR','IF','EIF'],
        index = outlierIxs
    )

    # fill nulls based on index value match
    outlierSummary['IQR'] = outlierSummary['IQR'].loc[iqrOutliers].fillna(value='X')
    outlierSummary['IF'] = outlierSummary['IF'].loc[ifOutliers].fillna(value='X')
    outlierSummary['EIF'] = outlierSummary['EIF'].loc[eifOutliers].fillna(value='X')

    # add summary columns and sort
    outlierSummary['Count'] = outlierSummary.count(axis = 1)
    outlierSummary = outlierSummary.sort_values(['Count'], ascending = False)

    outlierSummary = outlierSummary.fillna('')
    return outlierSummary