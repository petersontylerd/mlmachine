import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator

from collections import Counter

import eif


class OutlierIQR(TransformerMixin, BaseEstimator):
    """
    Documentation:

        ---
        Description:
            Identifies outliers using inter quartile range (IQR) method using object
            oriented approach.

        ---
        Parameters:
            outlier_count : int
                Minimum number of values across all features for an single observation that
                need to be outliers in order for an observation to be flagged.
            iqr_step : float
                Multiplier that controls level of sensitivity of outlier detection method.
                Higher values for iqr_step will cause OutlierIQR to only detect increasingly
                extreme values.
            features : list
                List of features to be evaluated for outliers.
            drop_outliers : bool, default=False
                If True, drops outliers from the input data.

        ---
        Returns:
            x : array
                Dataset with outlier observations removed.
    """

    def __init__(self, outlier_count, iqr_step, features, drop_outliers=False):
        self.outlier_count = outlier_count
        self.iqr_step = iqr_step
        self.features = features
        self.drop_outliers = drop_outliers

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # empty list for holding index values for observations flagged as outlier
        outlier_indices = []

        # iterate over features
        for col in self.features:

            # calculate 25th percentile, 75th percentile and IQR
            q1 = np.percentile(X[col], 25)
            q3 = np.percentile(X[col], 75)
            iqr = q3 - q1

            # apply outlier step multiplier
            outlier_step = self.iqr_step * iqr

            # identify values that are higher or lower than the IQR
            outlier_list_col = X[
                (X[col] < q1 - outlier_step) | (X[col] > q3 + outlier_step)
            ].index

            # append the outlier observation index values
            outlier_indices.extend(outlier_list_col)

        ## identify observations with >= outlier_count outlier values among the features
        # count number of times each observation index appears in outlier_indices
        outlier_indices = Counter(outlier_indices)

        # reduce outlier_indices to the observations with >= outlier_count outlier values
        self.outliers = list(
            k for k, v in outlier_indices.items() if v >= self.outlier_count
        )

        # optionally drop outlier observations from input dataset
        if self.drop_outliers:
            X = X.drop(self.outliers, axis=0).reset_index(drop=True)

        return X


class ExtendedIsoForest(TransformerMixin, BaseEstimator):
    """
    Documentation:

        ---
        Description:
            Identifies outliers using extended isolation forest method.

        ---
        Parameters:
            columns : list
                Features to be evaluate.
            n_trees : int
                Number of trees used.
            sample_size : int
                Sub_sample size for creating each trees. Values must be smaller that input dataset
                row count.
            ExtensionLevel : int
                Degrees of freedom for choosing hyperplanes that divide data. Value must be smaller
                than input dataset column count.
            anomalies_ratio : float
                Percent of input dataset observations to identify as outliers.
            drop_outliers : bool, default=False
                Dictates whether identified outliers are removed from input dataset.

        ---
        Returns:
            x : array
                Dataset with outlier observations removed.
    """

    def __init__(self, columns, n_trees, sample_size, extension_level, anomalies_ratio, drop_outliers=False):
        self.columns = columns
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.extension_level = extension_level
        self.anomalies_ratio = anomalies_ratio
        self.drop_outliers = drop_outliers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # instantiate extended isolation forest object
        ext_iso = eif.iForest(
            X=X[self.columns].values,
            ntrees=self.n_trees,
            sample_size=self.sample_size,
            ExtensionLevel=self.extension_level,
        )

        # calculate anomaly scores
        anomaly_scores = ext_iso.compute_paths(X_in=X[self.columns].values)

        # store anomaly score for each observation in Pandas DataFrame, sort descending
        anomaly_scores_sorted = pd.DataFrame(
            anomaly_scores, index=X.index, columns=["anomaly score"]
        ).sort_values(["anomaly score"], ascending=False)

        # identify outliers by limiting anomaly_scores_sorted to the (anamalies_ratio * total observation)
        # observations with the highest anomaly scores
        self.outliers = np.array(
            anomaly_scores_sorted[
                : int(np.ceil(self.anomalies_ratio * X.shape[0]))
            ].index
        )

        # optionally drop outlier observations from input dataset
        if self.drop_outliers:
            X = X.drop(self.outliers, axis=0).reset_index(drop=True)

        return X

def outlier_summary(self, iqr_outliers, if_outliers, eif_outliers):
    """
    Documentation:

        ---
        Description:
            Creates Pandas DataFrame summarizing which observations were flagged
            as outliers and by which outlier detection method each observation was
            identified.

        ---
        Parameters:
            iqr_outliers : array
                Array contains indexes of observations identified as outliers using
                IQR method.
            if_outliers : array
                Array contains indexes of observations identified as outliers using
                isolition forest method.
            eif_outliers : array
                Array contains indexes of observations identified as outliers using
                extended isolition forest method.

        ---
        Returns:
            outlier_summary : Pandas DataFrame
                DataFrame summarizing outlier detection results
    """
    # merge and de_duplicate outlier index values
    outlier_ixs = np.unique(np.concatenate([iqr_outliers, if_outliers, eif_outliers]))

    # create shell dataframe
    outlier_summary = pd.DataFrame(columns=["iqr", "if", "eif"], index=outlier_ixs)

    # fill nulls based on index value match
    outlier_summary["iqr"] = outlier_summary["iqr"].loc[iqr_outliers].fillna(value="x")
    outlier_summary["if"] = outlier_summary["if"].loc[if_outliers].fillna(value="x")
    outlier_summary["eif"] = outlier_summary["eif"].loc[eif_outliers].fillna(value="x")

    # add summary columns and sort
    outlier_summary["count"] = outlier_summary.count(axis=1)
    outlier_summary = outlier_summary.sort_values(["count"], ascending=False)

    outlier_summary = outlier_summary.fillna("")
    return outlier_summary

def outlier_IQR(self, data, iqr_step):
    """
    Documentation:

        ---
        Description:
            Identifies outliers using inter quartile range (IQR) method using functional
            approach.

        ---
        Parameters:
            data : Pandas Series
                Input data array.
            iqr_step : float
                Multiplier that controls level of sensitivity of outlier detection method.
                Higher values for iqr_step will cause OutlierIQR to only detect increasingly
                extreme values.

        ---
        Returns:
            outlier_index : array
                Index of outliers in original data array.
    """
    # calculate 25th percentile, 75th percentile and IQR
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # apply outlier step multiplier
    outlier_step = iqr_step * iqr

    # identify values that are higher or lower than the IQR
    outlier_index = data[(data < q1 - outlier_step) | (data > q3 + outlier_step)].index

    return outlier_index