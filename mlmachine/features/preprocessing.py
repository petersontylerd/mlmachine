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
import collections
import copy


class ContextImputer(base.TransformerMixin, base.BaseEstimator):
    """
    documentation:
        description:
            impute numberal columns with certain value, as specified by the strategy parameter. also
            utilizes one or more additional context columns as a group by value to add more subtlety to
            fill_value identification. imputes training data features, and stores impute values to be used
            on validation and unseen data.
        parameters:
            null_col : list
                column with nulls to be imputed.
            context_col : list
                list of one or most columns to group by to add context to null column.
            strategy : string, default = 'mean'
                imputing stategy. takes values 'mean', 'median' and 'most_frequent'.
            train : bool, default=True
                tells class whether we are imputing training data or unseen data.
            train_value : dict, default=None
                only used when train=False. value is a dictionary containing 'feature : value' pairs.
                dictionary is retrieved from training data pipeline using named steps. the attribute is
                called train_value_.
        returns:
            X : array
                dataset where each column with missing values has been imputed with a value learned from a particular
                strategy while also consider select columns as a group by variable.
    """

    def __init__(
        self, null_col, context_col, strategy="mean", train=True, train_value=None
    ):
        self.null_col = null_col
        self.context_col = context_col
        self.strategy = strategy
        self.train = train
        self.train_value = train_value

    def fit(self, X, y=None):
        if self.strategy == "mean":
            self.train_value = (
                X[X[self.null_col].notnull()]
                .groupby(self.context_col)
                .mean()[self.null_col]
            )
        elif self.strategy == "median":
            self.train_value = (
                X[X[self.null_col].notnull()]
                .groupby(self.context_col)
                .median()[self.null_col]
            )
        elif self.strategy == "most_frequent":
            self.train_value = (
                X[X[self.null_col].notnull()]
                .groupby(self.context_col)[self.null_col]
                .agg(lambda X: X.value_counts().index[0])
            )

        self.train_value = self.train_value.reset_index()
        return self

    def transform(self, X):
        # impute missing values based on train_value
        if isinstance(self.context_col, str):
            X[self.null_col] = np.where(
                X[self.null_col].isnull(),
                X[self.context_col].map(
                    self.train_value.set_index(self.context_col)[self.null_col]
                ),
                X[self.null_col],
            )
        return X[self.null_col]


class DataFrameSelector(base.BaseEstimator, base.TransformerMixin):
    """
    documentation:
        description:
            select a susbset set of features of a pandas DataFrame as part of a
            pipeline. capable of select and/or deselecting columns by name and by
            data type.

            Note - if there is a logical conflict between include and exclude
            parameters, the class will first prioritize the column parameters
            over the dtype parameters in order to support subsetting. If the
            logic cannot be resolved by this rule alone, exclusion parameters
            will be prioritized over inclusion parameters.

        parameters:
            include_columns : list
                list of features to select from Pandas DataFrame.
            include_pd_dtypes : list
                list of strings describing pandas dtypes to select.
            include_mlm_dtypes : list
                list of strings describing mlmachine dtypes to select.
            exclude_columns : list
                list of features to deselect from Pandas DataFrame.
            exclude_pd_dtypes : list
                list of strings describing pandas dtypes to deselect.
            exclude_mlm_dtypes : list
                list of strings describing mlmachine dtypes to select.
    """

    def __init__(self, include_columns=None, include_pd_dtypes=None, include_mlm_dtypes=None,
                    exclude_columns=None, exclude_pd_dtypes=None, exclude_mlm_dtypes=None):

        self.include_columns = include_columns
        self.include_pd_dtypes = include_pd_dtypes
        self.include_mlm_dtypes = include_mlm_dtypes
        self.exclude_columns = exclude_columns
        self.exclude_pd_dtypes = exclude_pd_dtypes
        self.exclude_mlm_dtypes = exclude_mlm_dtypes

    def fit(self, X, y=None):

        # preserve feature_by_mlm_dtype if it exists
        try:
            self.meta = X.feature_by_mlm_dtype
            self.no_meta = False
        except AttributeError:
            if self.include_mlm_dtypes is not None or self.exclude_mlm_dtypes is not None:
                raise AttributeError ("Attempting to filter using mlm dtypes, but feature_by_mlm_dtype object is not associated with the input Pandas DataFrame.")
            else:
                pass

        self.all_columns = X.columns.tolist()
        self.selected_columns = []
        self.remove_columns = []

        ## selection
        # select columns by name
        if self.include_columns is not None:
            self.selected_columns.extend(self.include_columns)

        # select columns by pandas dtype
        if self.include_pd_dtypes is not None:
            for dtype in self.include_pd_dtypes:
                self.selected_columns.extend(
                    X.select_dtypes(include=dtype).columns.tolist()
                )

        # select columns by mlmachine dtype
        if self.include_mlm_dtypes is not None:
            for dtype in self.include_mlm_dtypes:
                self.selected_columns.extend(X.feature_by_mlm_dtype[dtype])

        # flatten list and remove duplicates
        self.selected_columns = collections.OrderedDict.fromkeys(self.selected_columns)
        self.selected_columns = list(self.selected_columns.keys())

        ## deselection
        # deselect columns by name
        if self.exclude_columns is not None:
            self.remove_columns.extend(self.exclude_columns)

        # deselect columns by pandas dtype
        if self.exclude_pd_dtypes is not None:
            for dtype in self.exclude_pd_dtypes:
                self.remove_columns.extend(
                    X.select_dtypes(include=dtype).columns.tolist()
                )

        # deselect columns by pandas dtype
        if self.exclude_mlm_dtypes is not None:
            for dtype in self.exclude_mlm_dtypes:
                self.remove_columns.extend(X.feature_by_mlm_dtype[dtype])

        # flatten list and remove duplicates
        self.remove_columns = list(set(self.remove_columns))

        ## reconcile selections and removals
        # if the only input is to exclude, remove remove_columns from all_columns
        if len(self.remove_columns) > 0 and len(self.selected_columns) == 0:
            self.final_columns = list(
                set(self.all_columns).difference(self.remove_columns)
            )

        # if the only input is to include, keep only the select_columns
        elif len(self.selected_columns) > 0 and len(self.remove_columns) == 0:
            self.final_columns = self.selected_columns

        #
        elif (
            self.include_columns is not None
            and (self.include_pd_dtypes is None and self.include_mlm_dtypes is None)
            and self.exclude_columns is not None
            and (self.exclude_pd_dtypes is None and self.exclude_mlm_dtypes is None)
        ):
            self.final_columns = self.selected_columns
            self.final_columns = [column for column in self.final_columns if column not in self.remove_columns]

        #
        elif (
            self.include_columns is not None
            and (self.include_pd_dtypes is None and self.include_mlm_dtypes is None)
            and self.exclude_columns is None
            and (self.exclude_pd_dtypes is not None or self.exclude_mlm_dtypes is not None)
        ):
            self.final_columns = self.selected_columns

        #
        elif (
            self.include_columns is None
            and (self.include_pd_dtypes is not None or self.include_mlm_dtypes is not None)
            and self.exclude_columns is not None
            and (self.exclude_pd_dtypes is None and self.exclude_mlm_dtypes is None)
        ):
            self.final_columns = self.selected_columns
            self.final_columns = [column for column in self.final_columns if column not in self.remove_columns]

        #
        elif (
            self.include_columns is None
            and (self.include_pd_dtypes is not None or self.include_mlm_dtypes is not None)
            and self.exclude_columns is None
            and (self.exclude_pd_dtypes is not None or self.exclude_mlm_dtypes is not None)
        ):
            self.final_columns = self.selected_columns
            self.final_columns = [column for column in self.final_columns if column not in self.remove_columns]

        #
        elif (
            self.include_columns is not None
            and (self.include_pd_dtypes is not None or self.include_mlm_dtypes is not None)
            and self.exclude_columns is not None
            and (self.exclude_pd_dtypes is None and self.exclude_mlm_dtypes is None)
        ):
            self.final_columns = self.selected_columns
            self.final_columns = [column for column in self.final_columns if column not in self.remove_columns]

        #
        elif (
            self.include_columns is not None
            and (self.include_pd_dtypes is not None or self.include_mlm_dtypes is not None)
            and self.exclude_columns is None
            and (self.exclude_pd_dtypes is not None or self.exclude_mlm_dtypes is not None)
        ):
            self.final_columns = self.selected_columns

        #
        elif (
            self.include_columns is None
            and (self.include_pd_dtypes is not None or self.include_mlm_dtypes is not None)
            and self.exclude_columns is not None
            and (self.exclude_pd_dtypes is not None or self.exclude_mlm_dtypes is not None)
        ):
            self.final_columns = self.selected_columns
            self.final_columns = [column for column in self.final_columns if column not in self.remove_columns]

        #
        elif (
            self.include_columns is not None
            and (self.include_pd_dtypes is None and self.include_mlm_dtypes is None)
            and self.exclude_columns is not None
            and (self.exclude_pd_dtypes is not None or self.exclude_mlm_dtypes is not None)
        ):
            self.final_columns = self.selected_columns

        #
        elif (
            self.include_columns is not None
            and (self.include_pd_dtypes is not None or self.include_mlm_dtypes is not None)
            and self.exclude_columns is not None
            and (self.exclude_pd_dtypes is not None or self.exclude_mlm_dtypes is not None)
        ):
            self.final_columns = self.selected_columns

        #
        elif (
            self.include_columns is None
            and self.include_pd_dtypes is None
            and self.include_mlm_dtypes is None
            and self.exclude_columns is None
            and self.exclude_pd_dtypes is None
            and self.exclude_mlm_dtypes is None
        ):
            self.final_columns = self.all_columns

        return self

    def transform(self, X):
        return X[self.final_columns]


class PandasPipeline(base.TransformerMixin, base.BaseEstimator):
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
        self.original_columns = X.columns
        self.dtypes = X[self.original_columns].dtypes

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
            for a,b in zip(self.original_columns, prefixes):
                names = list(map(lambda x: str.replace(x, b, a + "_"), names))
            self.original_columns = names

        # if the class is a PolynomialFeatures instance
        elif isinstance(self.est, preprocessing._data.PolynomialFeatures):

            ### replace feature code names with actual feature names
            # capture object's feature code names
            names = self.est.get_feature_names()

            # remove "x" prefix from features code names
            feature_code_names = list(map(lambda x:str.replace(x, "x", ""), names))

            # reduce feature_code_names down to [0,1,2,3...] by ignoring "^2" and sort as integers
            feature_code_names = sorted(list(set([x.replace("^2","").split(" ")[0] for x in feature_code_names])), key=int)

            # add "x" as a prefix and an "_" as a suffix to the
            feature_code_names = ["x" + x +"_" for x in feature_code_names]

            ### build actual feature names
            # replace " " with "_ " and add an additional "_" to the full string yield names such as "x0_ x1_"
            feature_actual_names = [x.replace(" ","_ ") + "_" for x in names]

            # change strings such as "x02_" to yield names such as "x0_2"
            feature_actual_names = [x.replace("^2_","_^2") for x in feature_actual_names]

            # change strings such as "x0_ x1_" to yield names such as "x0_*x1_"
            feature_actual_names = [x.replace(" ","*") for x in feature_actual_names]

            # replace code names with actual column names
            for a,b in zip(self.original_columns, feature_code_names):
                feature_actual_names = list(map(lambda x: str.replace(x, b, a), feature_actual_names))

            self.original_columns = feature_actual_names

        # if the class is a KBinsDiscretizer instance
        elif isinstance(self.est, preprocessing._discretization.KBinsDiscretizer):
            names = []
            bins = self.est.n_bins
            for col in self.original_columns:
                names.append(col + "_" + str(bins) + "_bins")

            self.original_columns = names

        # if the class is a OrdinalEncoder instance
        elif isinstance(self.est, preprocessing._encoders.OrdinalEncoder):
            names = []
            for col in self.original_columns:
                names.append(col + "_encoded")

            self.original_columns = names

        # if the class is a category_encoders CountEncoder instance
        elif isinstance(self.est, ce.count.CountEncoder):
            names = []

            for col in self.original_columns:
                names.append(col + "_count_encoded")

            self.original_columns = names

        # if the class is a category_encoders BinaryEncoder instance
        elif isinstance(self.est, ce.binary.BinaryEncoder):
            names = []
            self.original_columns = self.est.get_feature_names()

            for col in self.original_columns:
                names.append(col + "_binarized")

            self.original_columns = names

        # if the class is a category_encoders BinaryEncoder instance
        elif isinstance(self.est, preprocessing._data.QuantileTransformer):
            names = []

            for col in self.original_columns:
                names.append(col + "_quantile_" + self.est.output_distribution)

            self.original_columns = names
        return self

    def transform(self, X, y=None, copy=None):
        self.index = X.index

        X = self.est.transform(X)
        if sparse.issparse(X):
            X = X.todense()

        # create new Pandas DataFrame
        X = pd.DataFrame(X, columns=self.original_columns, index=self.index)

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

        # preserve feature_by_mlm_dtype if it exists
        try:
            self.meta = X.feature_by_mlm_dtype
            self.no_meta = False
        except AttributeError:
            self.no_meta = True
            pass

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

        if not self.no_meta:
            Xs = Xs.loc[:, ~Xs.columns.duplicated()]
            Xs = PreserveMetaData(Xs)
            Xs.feature_by_mlm_dtype = self.meta

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

        if not self.no_meta:
            Xs = Xs.loc[:, ~Xs.columns.duplicated()]
            Xs = PreserveMetaData(Xs)
            Xs.feature_by_mlm_dtype = self.meta

        return Xs

class KFoldSelectEncoder(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, target, cv, encoder):
        self.target = target
        self.cv = cv
        self.encoder = encoder
        self.transform_train = False

    def fit(self, X, y=None):
        self.columns = X.columns
        self.stats = {}

        # capture name of encoder for column name
        encoder_name = self.encoder.__module__.split(".")
        if encoder_name[0] == "category_encoders":
            self.column_suffix = "_" + encoder_name[1]

        return self

    def transform(self, X, y=None):
        if not self.transform_train:
            # combine input columns and target
            X = X.merge(self.target, left_index=True, right_index=True)

            # add empty columns to input dataset and set as number
            for column in self.columns:
                X[column + self.column_suffix] = np.nan
                X[column + self.column_suffix] = pd.to_numeric(X[column + self.column_suffix])

            # iterate through cv indices
            for train_ix, valid_ix in self.cv.split(X):
                x_train, x_valid = X.iloc[train_ix], X.iloc[valid_ix]

                for column in self.columns:
                    enc = self.encoder(cols=[column])

                    # fit on train
                    x_train[column + self.column_suffix] = enc.fit_transform(x_train[column], x_train[self.target.name])
                    x_valid[column + self.column_suffix] = enc.transform(x_valid[column])
                    X.loc[X.index[valid_ix], column + self.column_suffix] = x_valid[column + self.column_suffix].map(x_valid[column + self.column_suffix]).fillna(x_valid[column + self.column_suffix])

            # collect average values for transformation of unseen data
            for column in self.columns:
                self.stats[column] = X.groupby(column)[column + self.column_suffix].mean()

            # flip transform_train switch to indicate that fitting has occurred
            # ensure future transform calls will go to the else branch of the conditional
            self.transform_train = True

            # drop target column
            X = X.drop(self.target.name, axis=1)

        else:
            # iterate through all columns in the summary stats dict
            for column in self.stats.keys():

                # create empty colunn
                X[column + self.column_suffix] = np.nan
                X[column + self.column_suffix] = pd.to_numeric(X[column + self.column_suffix])

                # fill in empty column with mean values grouped by target
                X[column + self.column_suffix] = np.where(
                    X[column + self.column_suffix].isnull(),
                    X[column].map(self.stats[column]),
                    X[column + self.column_suffix],
                )
        return X

class DualTransformer(base.TransformerMixin, base.BaseEstimator):
    """
    documentation:
        description:
            performs yeo-johnson transformation on all specified feature. also performs box_cox transformation. if
            the minimum value in a feature is 0, box_cox + 1 transformation is performed. if minimum value is greater
            than 0, standard box_cox transformation is performed. each transformation process automatically determines
            the optimal lambda value. these values are stored for transformation of validation data.

            note - this method adds additional columns rather than updating existing columns inplace.
        parameters:
            yeojohnson : bool, default=True
                conditional controls whether yeo-johnson transformation is applied to input data.
            boxcox : bool, default=True
                conditional controls whether box_cox transformation is applied to input data.
        returns:
            X : array
                yeo-johnson, and potentially box_cox (or box_cox + 1) transformed input data.
    """

    def __init__(self, yeojohnson=True, boxcox=True):
        self.yeojohnson = yeojohnson
        self.boxcox = boxcox

    def fit(self, X, y=None):
        self.original_columns = X.columns

        ## yeo-johnson
        if self.yeojohnson:

            self.yj_lambdas_dict_ = {}

            # collect lambas using sklearn implementation
            yj = preprocessing.PowerTransformer(method="yeo-johnson")
            yj_lambdas = yj.fit(X).lambdas_

            # cycle through columns, add transformed columns, store lambdas
            for ix, col in enumerate(X.columns):
                self.yj_lambdas_dict_[col] = yj_lambdas[ix]

        ## box_cox
        if self.boxcox:

            self.bc_p1_lambdas_dict_ = {}
            self.bc_lambdas_dict_ = {}

            for col in X.columns:
                # if minimum feature value is 0, do box_cox + 1
                if np.min(X[col]) == 0:
                    _, lmbda = stats.boxcox(X[col].values + 1, lmbda=None)
                    self.bc_p1_lambdas_dict_[col] = lmbda
                # otherwise do standard box_cox
                elif np.min(X[col]) > 0:
                    _, lmbda = stats.boxcox(X[col].values, lmbda=None)
                    self.bc_lambdas_dict_[col] = lmbda
        return self

    def transform(self, X):
        # yeo-johnson
        if self.yeojohnson:
            for col in self.yj_lambdas_dict_.keys():
                X[col + "YeoJohnson"] = stats.yeojohnson(
                    X[col].values, lmbda=self.yj_lambdas_dict_[col]
                )

        # box_cox
        if self.boxcox:
            for col in self.bc_p1_lambdas_dict_.keys():
                X[col + "BoxCox"] = stats.boxcox(
                    X[col].values + 1, lmbda=self.bc_p1_lambdas_dict_[col]
                )

            for col in self.bc_lambdas_dict_.keys():
                X[col + "BoxCox"] = stats.boxcox(
                    X[col].values, lmbda=self.bc_lambdas_dict_[col]
                )
        return X
        # return X.drop(self.original_columns, axis=1)

def skew_summary(self, data=None, columns=None):
    """
    documentation:
        description:
            displays pandas DataFrame summarizing the skew of each number variable. also summarizes
            the percent of a column that has a value of zero.
        parameters:
            data : pandas DataFrame, default=None
                pandas DataFrame containing independent variables. if left as none,
                the feature dataset provided to machine during instantiation is used.
            columns : list of strings, default=None
                list containing string names of columns. if left as none, the value associated
                with sel.feature_by_mlm_dtype["number"] will be used as the column list.
    """
    # use data/feature_by_mlm_dtype["number"] columns provided during instantiation if left unspecified
    if data is None:
        data = self.data
    if columns is None:
        columns = self.data.feature_by_mlm_dtype["continuous"]

    skewness = (
        data[columns]
        .apply(lambda X: stats.skew(X.dropna()))
        .sort_values(ascending=False)
    )
    skewness = pd.DataFrame({"skew": skewness})

    # add column describing percent of values that are zero
    skewness["pct_zero"] = np.nan
    for col in columns:

        try:
            skewness.loc[col]["pct_zero"] = data[data[col] == 0][
                col
            ].value_counts() / len(data)
        except ValueError:
            skewness.loc[col]["pct_zero"] = 0.0
    skewness = skewness.sort_values(["skew"])
    return skewness

class PreserveMetaData(pd.DataFrame):

    _metadata = ["feature_by_mlm_dtype"]

    @property
    def _constructor(self):
        return PreserveMetaData