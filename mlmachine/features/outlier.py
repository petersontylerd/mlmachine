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
        anomalyScoresSorted = np.argsort(anomalyScores)
        self.outliers_ = np.array(
            anomalyScoresSorted[-int(np.ceil(self.anomaliesRatio * X.shape[0])):]
        )

        if self.dropOutliers:
            X = X.drop(self.outliers_, axis=0).reset_index(drop=True)

        return X