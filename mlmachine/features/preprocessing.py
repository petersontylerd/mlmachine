import numpy as np
import pandas as pd

from scipy import sparse, stats, special

import sklearn.base as base
import sklearn.impute as impute
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing
from sklearn.externals.joblib import Parallel, delayed

import category_encoders as ce

import itertools
import copy


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
        # capture original column names and dtypes
        self.originalColumns = X.columns
        self.dtypes = X[self.originalColumns].dtypes

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

        # if the class is a PolynomialFeatures instance
        elif isinstance(self.est, preprocessing.data.PolynomialFeatures):

            ### replace feature code names with actual feature names
            # capture object's feature code names
            names = self.est.get_feature_names()

            # remove "x" prefix from features code names
            featureCodeNames = list(map(lambda x:str.replace(x, "x", ""), names))

            # reduce featureCodeNames down to [0,1,2,3...] by ignoring "^2" and sort as integers
            featureCodeNames = sorted(list(set([x.replace("^2","").split(" ")[0] for x in featureCodeNames])), key=int)

            # add "x" as a prefix and an "_" as a suffix to the
            featureCodeNames = ["x" + x +"_" for x in featureCodeNames]

            ### build actual feature names
            # replace " " with "_ " and add an additional "_" to the full string yield names such as "x0_ x1_"
            featureActualNames = [x.replace(" ","_ ") + "_" for x in names]

            # change strings such as "x02_" to yield names such as "x0_2"
            featureActualNames = [x.replace("^2_","_^2") for x in featureActualNames]

            # change strings such as "x0_ x1_" to yield names such as "x0_*x1_"
            featureActualNames = [x.replace(" ","*") for x in featureActualNames]

            # replace code names with actual column names
            for a,b in zip(self.originalColumns, featureCodeNames):
                featureActualNames = list(map(lambda x: str.replace(x, b, a), featureActualNames))

            self.originalColumns = featureActualNames

        # if the class is a KBinsDiscretizer instance
        elif isinstance(self.est, preprocessing._discretization.KBinsDiscretizer):
            names = []
            bins = self.est.n_bins
            for col in self.originalColumns:
                names.append(col + "_" + str(bins) + "_Bins")

            self.originalColumns = names

        # if the class is a category_encoders CountEncoder instance
        elif isinstance(self.est, ce.count.CountEncoder):
            names = []

            for col in self.originalColumns:
                names.append(col + "_Count")

            self.originalColumns = names

        # if the class is a category_encoders BinaryEncoder instance
        elif isinstance(self.est, ce.binary.BinaryEncoder):
            names = []
            self.originalColumns = self.est.get_feature_names()

            for col in self.originalColumns:
                names.append(col + "_Binarized")

            self.originalColumns = names
        return self

    def transform(self, X, y=None, copy=None):
        self.index = X.index

        X = self.est.transform(X)
        if sparse.issparse(X):
            X = X.todense()

        # create new Pandas DataFrame
        X = pd.DataFrame(X, columns=self.originalColumns, index=self.index)

        return X


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


class UnprocessedColumnAdder(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Add the columns not processed by other transformers in pipeline to the final
            output dataset.
        Parameters:
            pipes : sklearn FeatureUnion pipeline object
                FeatureUnion pipeline object with all column-specific transformations.
    """
    def __init__(self, pipes):
        self.pipes = pipes
        self.processedColumns = []

    def fit(self, X, y=None):

        # for each transformer in the input pipe
        for pipeTrans in self.pipes.transformer_list:
            pipeCols = []

            # for each step in each transformer
            for step in pipeTrans[1].steps:
                # add all columns specified in DataFrameSelector objects
                if isinstance(step[1], DataFrameSelector):
                    pipeCols.append(step[1].attributeNames)
                    pipeCols = list(set(itertools.chain(*pipeCols)))
                # remove all columns specified in ContextImputer objects
                elif isinstance(step[1], ContextImputer):
                    removeCols = step[1].contextCol
                    if isinstance(removeCols, str):
                        pipeCols = [i for i in pipeCols if i not in [removeCols]]

            # collect columns in processedColumns attribute
            self.processedColumns.append(pipeCols)

        # flatten processedColumns
        self.processedColumns = list(set(itertools.chain(*self.processedColumns)))

        # compare columns of input dataset with processedColumns to identify unprocessed columns
        self.unprocessedColumns = list(set(X.columns) - set(self.processedColumns))
        return self

    def transform(self, X):
        return X[self.unprocessedColumns]


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
        return X
        # return X.drop(self.originalColumns, axis=1)


def skewSummary(self, data=None, columns=None):
    """
    Documentation:
        Description:
            Displays Pandas DataFrame summarizing the skew of each numeric variable. Also summarizes
            the percent of a column that has a value of zero.
        Parameters:
            data : Pandas DataFrame, default = None
                Pandas DataFrame containing independent variables. If left as None,
                the feature dataset provided to Machine during instantiation is used.
            columns : list of strings, default = None
                List containing string names of columns. If left as None, the value associated
                with sel.featureType["numeric"] will be used as the column list.
    """
    # use data/featureType["numeric"] columns provided during instantiation if left unspecified
    if data is None:
        data = self.data
    if columns is None:
        columns = self.featureType["numeric"]

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