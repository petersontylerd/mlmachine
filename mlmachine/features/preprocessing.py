import numpy as np
import pandas as pd

from scipy import sparse, stats, special

import sklearn.base as base
import sklearn.impute as impute
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing
from sklearn.externals.joblib import Parallel, delayed

import itertools
import copy

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


class ConvertToCategory(base.BaseEstimator, base.TransformerMixin):
    """
    Documentation:
        Description:
            Convert all specified columns in a Pandas DataFrame to the data
            type 'category'.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[X.columns] = X[X.columns].astype("category")
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
        if self.strategy == "mean":
            self.trainValue = X[X[self.nullCol].notnull()].groupby(self.contextCol).mean()[self.nullCol]
        elif self.strategy == "median":
            self.trainValue = X[X[self.nullCol].notnull()].groupby(self.contextCol).median()[self.nullCol]
        elif self.strategy == "most_frequent":
            self.trainValue = X[X[self.nullCol].notnull()].groupby(self.contextCol)[self.nullCol].agg(
                lambda x: x.value_counts().index[0]
            )

        self.trainValue = self.trainValue.reset_index()
        return self

    def transform(self, X):
        # impute missing values based on trainValue
        if isinstance(self.contextCol, str):
            X[self.nullCol] = np.where(
                X[self.nullCol].isnull(),
                X[self.contextCol].map(
                    self.trainValue.set_index(self.contextCol)[self.nullCol]
                ),
                X[self.nullCol],
            )
        return X[self.nullCol]


class DataFrameSelector(base.BaseEstimator, base.TransformerMixin):
    """
    Documentation:
        Description:
            Select a susbset set of features of a Pandas DataFrame as part of a
            pipeline.
        Parameters:
            attributeNames : list
                List of features to select from DataFrame.
    """
    def __init__(self, attributeNames):
        self.attributeNames = attributeNames

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attributeNames]


class PlayWithPandas(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Wrapper that ensures sklearn transformers will play nicely with Pandas.
            Pipelines will output a Pandas DataFrame with column names and original
            index values rather than a two dimensional numpy array.
        Parameters:
            attributeNames : list
                List of features to select from DataFrame to be passed into the
                wrapped transformer..
    """
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        # capture original column names and row index
        self.originalColumns = X.columns

        # create object containing fitted class
        self.est = self.transformer.fit(X)

        # if the class is a OneHotEncoder instance
        if isinstance(self.est, preprocessing._encoders.OneHotEncoder):
            names = self.est.get_feature_names()

            prefixes = list(map(lambda x:str.replace(x, "x", ""), names))
            prefixes = sorted(list(set([x.split("_")[0] for x in prefixes])), key=int)
            prefixes = ["x" + x + "_" for x in prefixes]

            # repalce "x0_, x1_..." prefix with original column anme
            # prefixes = sorted(list(set([x.split("_")[0] for x in names])))
            for a,b in zip(self.originalColumns, prefixes):
                names = list(map(lambda x: str.replace(x, b, a + "_"), names))
            self.originalColumns = names
        elif isinstance(self.est, preprocessing.data.PolynomialFeatures):
            names = self.est.get_feature_names()

            prefixes = list(map(lambda x:str.replace(x, "x", ""), names))
            prefixes = sorted(list(set([x.replace("^2","").split(" ")[0] for x in prefixes])), key=int)
            prefixes = ["x" + x +"_" for x in prefixes]

            names = [x.replace(" ","_ ") + "_" for x in names]
            names = [x.replace("^2_","_^2") for x in names]

            # repalce "x0_, x1_..." prefix with original column anme
            # prefixes = sorted(list(set([x[:2] for x in names])))

            for a,b in zip(self.originalColumns, prefixes):
                names = list(map(lambda x: str.replace(x, b, a), names))

            # names = list(map(lambda x:str.replace(x, " ", "*"), names))
            self.originalColumns = names
        return self

    def transform(self, X, y=None, copy=None):
        self.index = X.index

        X = self.est.transform(X)
        if sparse.issparse(X):
            X = X.todense()

        return pd.DataFrame(X, columns=self.originalColumns, index=self.index)


class PandasFeatureUnion(pipeline.FeatureUnion):
    """
    Documentation:
        Description:
            Modified version of sklearn's FeatureUnion class that outputs a Pandas DataFrame rather
            than a two dimensional array. Credit to X for this idea. Please take a look at the creator's
            original article detailing this function:

            https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union/index.html
    """

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(pipeline._fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(pipeline._transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs


class UnprocessedColumnAdder_old(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Identifies which columns were processed and which columns were not processed, and combines
            the updated processed columns with the unprocessed columns as they appear in the originalData
            parameter.
        Parameters:
            originalData : Pandas DataFrame
                Current state of dataset.
        Returns:
            X : Pandas DataFrame
                Pandas DataFrame containing merged processed columns and unprocessed columns.
    """
    def __init__(self, originalData):
        self.originalData = originalData

    def fit(self, X, y=None):
        self.processedColumns = X.columns
        self.unprocessedColumns = list(set(self.originalData.columns) - set(self.processedColumns))
        return self

    def transform(self, X):
        X = X.merge(self.originalData[self.unprocessedColumns], left_index=True, right_index=True)
        return X


class DualTransformer(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Performs Yeo-Johnson transformation on all specified feature. Also performs Box-Cox transformation. If
            the minimum value in a feature is 0, Box-Cox + 1 transformation is performed. If minimum value is greater
            than 0, standard Box-Cox transformation is performed. Each transformation process automatically determines
            the optimal lambda value. These values are stored for transformation of validation data.

            Note - this method adds additional columns rather than updating existing columns inplace.
        Parameters:
            yeojohnson : boolean, default = True
                Conditional controls whether Yeo-Johnson transformation is applied to input data.
            boxcox : boolean, default = True
                Conditional controls whether Box-Cox transformation is applied to input data.
        Returns:
            X : array
                Yeo-Johnson, and potentially Box-Cox (or Box-Cox + 1) transformed input data.
    """
    def __init__(self, yeojohnson=True, boxcox=True):
        self.yeojohnson = yeojohnson
        self.boxcox = boxcox

    def fit(self, X, y=None):
        self.originalColumns = X.columns

        ## Yeo-Johnson
        if self.yeojohnson:

            self.yjLambdasDict_ = {}

            # collect lambas using sklearn implementation
            yj = preprocessing.PowerTransformer(method="yeo-johnson")
            yjLambdas = yj.fit(X).lambdas_

            # cycle through columns, add transformed columns, store lambdas
            for ix, col in enumerate(X.columns):
                self.yjLambdasDict_[col] = yjLambdas[ix]

        ## Box-Cox
        if self.boxcox:

            self.bcP1LambdasDict_ = {}
            self.bcLambdasDict_ = {}

            for col in X.columns:
                # if minimum feature value is 0, do Box-Cox + 1
                if np.min(X[col]) == 0:
                    _, lmbda = stats.boxcox(X[col].values + 1, lmbda = None)
                    self.bcP1LambdasDict_[col] = lmbda
                # otherwise do standard Box-Cox
                elif np.min(X[col]) > 0:
                    _, lmbda = stats.boxcox(X[col].values, lmbda = None)
                    self.bcLambdasDict_[col] = lmbda
        return self

    def transform(self, X):
        # Yeo-Johnson
        if self.yeojohnson:
            for col in self.yjLambdasDict_.keys():
                X[col + "_yj"] = stats.yeojohnson(X[col].values, lmbda = self.yjLambdasDict_[col])

        # Box-Cox
        if self.boxcox:
            for col in self.bcP1LambdasDict_.keys():
                X[col + "_bc"] = stats.boxcox(X[col].values + 1, lmbda = self.bcP1LambdasDict_[col])

            for col in self.bcLambdasDict_.keys():
                X[col + "_bc"] = stats.boxcox(X[col].values, lmbda = self.bcLambdasDict_[col])
        return X.drop(self.originalColumns, axis=1)


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
    return skewness


def dataRefresh(transformeredTrainData, trainData, trainFeatureByDtype, transformeredValidationData=None, validationData=None, validFeatureByDtype=None, columnsToDrop=None):
    """
    Documentation:
        Description:
            Combine transformed data with the other unprocessed columns in the original dataset,
            while optionally removing columns that are no longer needed.
        Parameters:
            transformeredTrainData : Pandas DataFrame
                Output from data transformation pipeline representing the training data.
            trainData : Pandas DataFrame
                The training dataset as it currently exists.
            trainFeatureByDtype : dictionary
                Dictionary containing information on each feature's type (categorical, continuous, date).
            transformeredValidData : Pandas DataFrame, default = None
                Output from data transformation pipeline representing the training data.
            validData : Pandas DataFrame, default = None
                The validation dataset as it currently exists.
            validFeatureByDtype : dictionary, default - None
                Dictionary containing information on each feature's type (categorical, continuous, date).
            columnsToDrop : list, default = None
                Columns to drop from output dataset(s)/
        Returns:
            trainData : Pandas DataFrame
                The updated training dataset, which will include the processed columns and the unprocessed
                columns.
            trainFeatureByDtype : dictionary
                Dictionary containing information on each feature's type (categorical, continuous, date).
            validData : Pandas DataFrame
                The updated validation dataset, which will include the processed columns and the unprocessed
                columns. Optional, only gets returned when validation datset is provided.
            validFeatureByDtype : dictionary
                Dictionary containing information on each feature's type (categorical, continuous, date).
                Optional, only gets returned when validation datset is provided.

    """
    ###
    ## update tracked columns
    # gather all currently tracked columns
    allColumns = []
    for k in trainFeatureByDtype.keys():
        allColumns.append(trainFeatureByDtype[k])
    allColumns = list(set(itertools.chain(*allColumns)))

    # remove any columns listed in columnsToDrop
    if columnsToDrop is not None:
        # remove from allColumns
        allColumns = [x for x in allColumns if x not in columnsToDrop]

        # remove from featureByDtype
        for k in trainFeatureByDtype.keys():
            trainFeatureByDtype[k] = [x for x in trainFeatureByDtype[k] if x not in columnsToDrop]

    ###
    ## Training data
    # add unprocessed columns back to train.data
    trainDataUpdate = UnprocessedColumnAdder(originalData=trainData)
    trainData = trainDataUpdate.fit_transform(transformeredTrainData)

    # add new category columns to featureByDtype
    for col in trainData.select_dtypes(include="category").columns:
        try:
            if col not in allColumns and col not in columnsToDrop:
                trainFeatureByDtype["categorical"].append(col)
        # except if columnsToDrop is None
        except TypeError:
            if col not in allColumns:
                trainFeatureByDtype["categorical"].append(col)

    # add new numeric columns to featureByDtype
    for col in trainData.select_dtypes(exclude="category").columns:
        try:
            if col not in allColumns and col not in columnsToDrop:
                trainFeatureByDtype["continuous"].append(col)
        # except if columnsToDrop is None
        except TypeError:
            if col not in allColumns:
                trainFeatureByDtype["continuous"].append(col)
    # Drop
    if columnsToDrop is not None:
        trainData = trainData.drop(columnsToDrop, axis=1)

    ###
    ## Validation data
    if transformeredValidationData is not None:
        # add unprocessed column back to valid.data
        validDataUpdate = UnprocessedColumnAdder(originalData=validationData)
        validData = validDataUpdate.fit_transform(transformeredValidationData)

        if columnsToDrop is not None:
            validData = validData.drop(columnsToDrop, axis=1)

        # deep copy of train.featureByDtype
        validFeatureByDtype = copy.deepcopy(trainFeatureByDtype)

        return trainData, trainFeatureByDtype, validData, validFeatureByDtype
    else:
        return trainData, trainFeatureByDtype