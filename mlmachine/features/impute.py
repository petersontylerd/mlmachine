import numpy as np
import pandas as pd
import sklearn.preprocessing as prepocessing
import sklearn.base as base
import sklearn.impute as impute


class ModeImputer(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Impute columns with mode value. Imputes training data features, and stores
            mode values to be used on validation and unseen data.
        Parameters:
            cols : list
                List of features to be imputed.
            train : boolean, default = True
                Tells class whether we are imputing training data or unseen data.
            trainDict : dict, default = None
                Dictionary containing 'feature : mode value' pairs to be used to transform 
                validation data. Only used when train = False. Retrieved from training 
                data pipeline using named steps. Variable to be retrieved from traing 
                pipeline is called colValueDict_.
        Returns:
            X : array
                Dataset where each column with missing values has been imputed with the mode
                value of each particular columns.
    """

    def __init__(self, cols, train=True, trainDict=None):
        self.cols = cols
        self.train = train
        self.trainDict = trainDict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Encode training data
        if self.train:
            self.colValueDict_ = {}
            for col in self.cols:
                X[col] = X[col].fillna(X[col].value_counts().index[0])

                # build colValueDict
                self.colValueDict_[col] = X[col].value_counts().index[0]

        # For each columns, fill nulls with most frequently occuring value in training data
        else:
            for col in self.cols:
                X[col] = X[col].fillna(self.trainDict[col])
        return X


class NumericalImputer(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Impute numerical columns with certain value, as specified by the strategy 
            parameter. Imputes training data features, and stores imputed values to be 
            used on validation and unseen data.
        Parameters:
            cols : list
                List of features to be imputed.
            strategy : string, default = 'mean
                Imputing stategy. Takes values 'mean', 'median' and 'most_frequent'.
            train : boolean, default = True
                Tells class whether we are imputing training data or unseen data.
            trainDict : dict, default = None
                Dictionary containing 'feature : value' pairs to be used to transform 
                validation data. Only used when train = False. Retrieved from training 
                data pipeline using named steps. Variable to be retrieved from traing 
                pipeline is called colValueDict_.
        Returns:
            X : array
                Dataset where each column with missing values has been imputed with a value 
                learned from a particular strategy.
        """

    def __init__(self, cols, strategy="mean", train=True, trainDict=None):
        self.cols = cols
        self.strategy = strategy
        self.train = train
        self.trainDict = trainDict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Encode training data
        if self.train:
            self.colValueDict_ = {}
            for col in self.cols:
                imputed = impute.SimpleImputer(
                    missing_values=np.nan, strategy=self.strategy
                )
                X[col] = imputed.fit_transform(X[[col]])

                # build colValueDict
                self.colValueDict_[col] = imputed.statistics_

        # For each columns, fill nulls with most frequently occuring value in training data
        else:
            for col in self.cols:
                X[col] = X[col].fillna(self.trainDict[col][0])
        return X


class ConstantImputer(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Impute specified columns with a specific value. Intended to be used when a 
            missing value conveys that an observation does not have that attribute. If 
            column dtype is string, fill with 'Absent'. If column dtype is numerical, fill 
            with 0.
        Parameters:
            cols : list
                List of features to be imputed.
            fill : str, default = 'Absent'
                Default fill value.
        Returns:
            X : array
                Dataset where each column with missing values has been imputed the specified
                fill value.
    """

    def __init__(self, cols, fill="Absent"):
        self.cols = cols
        self.fill = fill

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.cols:
            imputed = impute.SimpleImputer(
                missing_values=np.nan, strategy="constant", fill_value=self.fill
            )
            X[col] = imputed.fit_transform(X[[col]])
        return X

class ContextImputer(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Impute numerical columns with certain value, as specified by the strategy parameter. Also
            utilizes one or more additional context columns as a group by value to add more subtlety to 
            fill-value identification. Imputes training data features, and stores impute values to be used 
            on validation and unseen data.
        Parameters:
            nullCol : list
                Column with nulls to be imputed.
            contextCol : list
                List of one or most columns to group by to add context to null column. 
            strategy : string, default = 'mean'
                Imputing stategy. Takes values 'mean', 'median' and 'most_frequent'.
            train : boolean, default = True
                Tells class whether we are imputing training data or unseen data.
            trainDf : dict, default = None
                Dataframe containing values to be mapped to replace nulls in validation data. Only used when 
                train = False. Retrieved from training data pipeline using named steps. Variable to be retrieved 
                from training pipeline is called fillDf.
        Returns:
            X : array
                Dataset where each column with missing values has been imputed with a value learned from a particular 
                strategy while also consider select columns as a group by variable.
    """

    def __init__(self, nullCol, contextCol, strategy="mean", train=True, trainDf=None):
        self.nullCol = nullCol
        self.contextCol = contextCol
        self.strategy = strategy
        self.train = train
        self.trainDf = trainDf

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Encode training data
        if self.train:
            if self.strategy == "mean":
                self.fillDf = X.groupby(self.contextCol).mean()[self.nullCol]
            elif self.strategy == "median":
                self.fillDf = X.groupby(self.contextCol).median()[self.nullCol]
            elif self.strategy == "most_frequent":
                self.fillDf = X.groupby(self.contextCol)[self.nullCol].agg(
                    lambda x: x.value_counts().index[0]
                )

            self.fillDf = self.fillDf.reset_index()

            X[self.nullCol] = np.where(
                X[self.nullCol].isnull(),
                X[self.contextCol].map(
                    self.fillDf.set_index(self.contextCol)[self.nullCol]
                ),
                X[self.nullCol],
            )

        # For each column, impute nulls with preferred value as determined in training data
        else:
            X[self.nullCol] = np.where(
                X[self.nullCol].isnull(),
                X[self.contextCol].map(
                    self.trainDf.set_index(self.contextCol)[self.nullCol]
                ),
                X[self.nullCol],
            )
        return X

