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
            trainValue : dict, default = None
                Only used when train = False. Either a dictionary containing 'feature : value' pairs
                or a Pandas DataFrame containing the full training dataset. Dictionary is retrieved from
                training data pipeline using named steps. The attribute is called trainValue_.

                DataFrame is used in instances when the feature to be imputed in the validation dataset
                did not have any missing values in the training dataset. This occurs in two circumstances:
                First, the feature was fully populated in the training dataset. Second, when following
                ContextImputer, the context value in the was fully populated in the training dataset
                but not the validation dataset.
        Returns:
            X : array
                Dataset where each column with missing values has been imputed with the mode
                value of each particular columns.
    """

    def __init__(self, cols, train=True, trainValue=None):
        self.cols = cols
        self.train = train
        self.trainValue = trainValue

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Encode training data
        if self.train:
            self.trainValue_ = {}
            for col in self.cols:
                X[col] = X[col].fillna(X[col].value_counts().index[0])

                # build colValueDict
                self.trainValue_[col] = X[col].value_counts().index[0]

        # For each columns, fill nulls with most frequently occuring value in training data
        else:
            # derive from dictionary learned from training data
            if isinstance(self.trainValue, dict):
                for col in self.cols:
                    X[col] = X[col].fillna(self.trainValue[col])

            # use the training dataset when learned data is not available
            elif isinstance(self.trainValue, pd.DataFrame):
                try:
                    self.trainValue = self.trainValue[self.cols].mode()
                except KeyError as e:
                    print(e)
                    print('\nModeImputer failed: the issue is likely that one or more of the specified columns was converted to a set of dummy columns in the training dataset, meaning the original column no longer exists in the training dataset.')
                    print('A potential solution is to pass this to trainValue:\n\t train.data.merge(trainPipe.named_steps["dummyNominal"].originalCols_, right_index = True, left_index = True)')
                    raise

                for col in self.cols:
                    X[col] = X[col].fillna(self.trainValue[col][0])
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
            trainValue : dict or Pandas DataFrame, default = None
                Only used when train = False. Either a dictionary containing 'feature : value' pairs
                or a Pandas DataFrame containing the full training dataset. Dictionary is retrieved from
                training data pipeline using named steps. The attribute is called trainValue_.

                DataFrame is used in instances when the feature to be imputed in the validation dataset
                did not have any missing values in the training dataset. This occurs in two circumstances:
                First, the feature was fully populated in the training dataset. Second, when following
                ContextImputer, the context value in the was fully populated in the training dataset
                but not the validation dataset.
        Returns:
            X : array
                Dataset where each column with missing values has been imputed with a value
                learned from a particular strategy.
        """

    def __init__(self, cols, strategy="mean", train=True, trainValue=None):
        self.cols = cols
        self.strategy = strategy
        self.train = train
        self.trainValue = trainValue

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Encode training data
        if self.train:
            self.trainValue_ = {}
            for col in self.cols:
                imputed = impute.SimpleImputer(
                    missing_values=np.nan, strategy=self.strategy
                )
                X[col] = imputed.fit_transform(X[[col]])

                # build colValueDict
                self.trainValue_[col] = imputed.statistics_
        # For each columns, fill nulls with most frequently occuring value in training data
        else:

            # derive from dictionary learned from training data
            if isinstance(self.trainValue, dict):
                for col in self.cols:
                    X[col] = X[col].fillna(self.trainValue[col][0])

            # use the training dataset when learned data is not available
            elif isinstance(self.trainValue, pd.DataFrame):
                if self.strategy == "mean":
                    self.trainValue = self.trainValue[self.cols].mean().to_frame().T
                elif self.strategy == "median":
                    self.trainValue = self.trainValue[self.cols].median().to_frame().T
                elif self.strategy == "most_frequent":
                    self.trainValue = self.trainValue[self.cols].mode()

                for col in self.cols:
                    X[col] = X[col].fillna(self.trainValue[col][0])
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
            trainValue : dict, default = None
                Only used when train = False. Value is a dictionary containing 'feature : value' pairs.
                Dictionary is retrieved from training data pipeline using named steps. The attribute is
                called trainValue_.
        Returns:
            X : array
                Dataset where each column with missing values has been imputed with a value learned from a particular
                strategy while also consider select columns as a group by variable.
    """

    def __init__(self, nullCol, contextCol, strategy="mean", train=True, trainValue=None):
        self.nullCol = nullCol
        self.contextCol = contextCol
        self.strategy = strategy
        self.train = train
        self.trainValue = trainValue

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode training data
        if self.train:
            if self.strategy == "mean":
                self.trainValue_ = X[X[self.nullCol].notnull()].groupby(self.contextCol).mean()[self.nullCol]
            elif self.strategy == "median":
                self.trainValue_ = X[X[self.nullCol].notnull()].groupby(self.contextCol).median()[self.nullCol]
            elif self.strategy == "most_frequent":
                self.trainValue_ = X[X[self.nullCol].notnull()].groupby(self.contextCol)[self.nullCol].agg(
                    lambda x: x.value_counts().index[0]
                )

            self.trainValue_ = self.trainValue_.reset_index()

            # impute missing values based on trainValue_
            X[self.nullCol] = np.where(
                X[self.nullCol].isnull(),
                X[self.contextCol].map(
                    self.trainValue_.set_index(self.contextCol)[self.nullCol]
                ),
                X[self.nullCol],
            )

        # for each column, impute nulls with preferred value as determined in training data
        else:
            if isinstance(self.trainValue, pd.DataFrame):
                # impute missing values based on trainValue_
                X[self.nullCol] = np.where(
                    X[self.nullCol].isnull(),
                    X[self.contextCol].map(
                        self.trainValue.set_index(self.contextCol)[self.nullCol]
                    ),
                    X[self.nullCol],
                )
        return X