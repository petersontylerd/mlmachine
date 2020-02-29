import numpy as np
import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    CategoricalDtype,
)

from scipy import sparse, stats, special

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import (
    make_pipeline,
    Pipeline,
    FeatureUnion,
    _fit_transform_one,
    _transform_one,
)

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    PolynomialFeatures,
    OrdinalEncoder,
    LabelEncoder,
    OneHotEncoder,
    KBinsDiscretizer,
    QuantileTransformer,
    PowerTransformer,
    MinMaxScaler,
    _encoders,
    _data,
    _discretization,
)
from sklearn.externals.joblib import Parallel, delayed

from category_encoders import WOEEncoder, TargetEncoder, CatBoostEncoder, BinaryEncoder, CountEncoder

import itertools
import collections
import copy


class ContextImputer(TransformerMixin, BaseEstimator):
    """
    Documentation:
        Description:
            impute numberal columns with certain value, as specified by the strategy parameter. also
            utilizes one or more additional context columns as a group by value to add more subtlety to
            fill_value identification. imputes training data features, and stores impute values to be used
            on validation and unseen data.
        Parameters:
            null_col : list
                column with nulls to be imputed.
            context_col : list
                list of one or most columns to group by to add context to null column.
            strategy : string, default='mean'
                imputing stategy. takes values 'mean', 'median' and 'most_frequent'.
            train : bool, default=True
                tells class whether we are imputing training data or unseen data.
            train_value : dict, default=None
                only used when train=False. value is a dictionary containing 'feature : value' pairs.
                dictionary is retrieved from training data pipeline using named steps. the attribute is
                called train_value_.
        Returns:
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

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Documentation:
        Description:
            select a susbset set of features of a Pandas DataFrame as part of a
            pipeline. capable of select and/or deselecting columns by name and by
            data type.

            Note - if there is a logical conflict between include and exclude
            parameters, the class will first prioritize the column parameters
            over the dtype parameters in order to support subsetting. If the
            logic cannot be resolved by this rule alone, exclusion parameters
            will be prioritized over inclusion parameters.

        Parameters:
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

        # preserve mlm_dtypes if it exists
        try:
            self.meta_mlm_dtypes = X.mlm_dtypes
            self.no_meta_mlm_dtypes = False
        except AttributeError:
            if self.include_mlm_dtypes is not None or self.exclude_mlm_dtypes is not None:
                raise AttributeError ("Attempting to filter using mlm dtypes, but mlm_dtypes object is not associated with the input Pandas DataFrame.")
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
                self.selected_columns.extend(X.mlm_dtypes[dtype])

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
                self.remove_columns.extend(X.mlm_dtypes[dtype])

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

class PandasTransformer(TransformerMixin, BaseEstimator):
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
        if isinstance(self.est, _encoders.OneHotEncoder):
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
        elif isinstance(self.est, _data.PolynomialFeatures):

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
        elif isinstance(self.est, KBinsDiscretizer):
            names = []
            bins = self.est.n_bins
            for col in self.original_columns:
                names.append(col + "_binned_" + str(bins))

            self.original_columns = names

        # if the class is a OrdinalEncoder instance
        elif isinstance(self.est, OrdinalEncoder):
            names = []
            for col in self.original_columns:
                names.append(col + "_ordinal_encoded")

            self.original_columns = names

        # if the class is a category_encoders CountEncoder instance
        elif isinstance(self.est, CountEncoder):
            names = []

            for col in self.original_columns:
                names.append(col + "_count_encoded")

            self.original_columns = names

        # if the class is a category_encoders BinaryEncoder instance
        elif isinstance(self.est, BinaryEncoder):
            names = []
            self.original_columns = self.est.get_feature_names()

            for col in self.original_columns:
                names.append(col + "_binary_encoded")

            self.original_columns = names

        # if the class is a category_encoders BinaryEncoder instance
        elif isinstance(self.est, QuantileTransformer):
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

class PandasFeatureUnion(FeatureUnion):
    """
    Documentation:
        Description:
            Modified version of sklearn's FeatureUnion class that outputs a Pandas DataFrame rather
            than a two dimensional array.
    """

    def fit_transform(self, X, y=None, **fit_params):

        # preserve mlm_dtypes if it exists
        try:
            self.meta_mlm_dtypes = X.mlm_dtypes
            self.no_meta_mlm_dtypes = False
        except AttributeError:
            self.no_meta_mlm_dtypes = True
            pass

        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
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

        if not self.no_meta_mlm_dtypes:
            Xs = Xs.loc[:, ~Xs.columns.duplicated()]
            Xs = PreserveMetaData(Xs)
            Xs.mlm_dtypes = self.meta_mlm_dtypes

            # reset dtype for any columns that were turned into object columns
            for mlm_dtype in Xs.mlm_dtypes.keys():
                for column in Xs.mlm_dtypes[mlm_dtype]:
                    try:
                        if is_object_dtype(Xs[column]):
                            if mlm_dtype == "boolean":
                                Xs[column] = Xs[column].astype("boolean")
                            elif mlm_dtype == "continuous":
                                Xs[column] = Xs[column].astype("float64")
                            elif mlm_dtype == "category":
                                Xs[column] = Xs[column].astype("category")
                            elif mlm_dtype == "count":
                                Xs[column] = Xs[column].astype("int64")
                            elif mlm_dtype == "date":
                                Xs[column] = Xs[column].astype("datetime64[ns]")
                            elif mlm_dtype == "nominal":
                                Xs[column] = Xs[column].astype("category")
                            elif mlm_dtype == "ordinal":
                                Xs[column] = Xs[column].astype("category")
                    except KeyError:
                        continue

        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
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

        # metadata attributes
        if not self.no_meta_mlm_dtypes:
            Xs = Xs.loc[:, ~Xs.columns.duplicated()]
            Xs = PreserveMetaData(Xs)
            Xs.mlm_dtypes = self.meta_mlm_dtypes

            # reset dtype for any columns that were turned into object columns
            for mlm_dtype in Xs.mlm_dtypes.keys():
                for column in Xs.mlm_dtypes[mlm_dtype]:
                    try:
                        if is_object_dtype(Xs[column]):
                            if mlm_dtype == "boolean":
                                Xs[column] = Xs[column].astype("boolean")
                            elif mlm_dtype == "continuous":
                                Xs[column] = Xs[column].astype("float64")
                            elif mlm_dtype == "category":
                                Xs[column] = Xs[column].astype("category")
                            elif mlm_dtype == "count":
                                Xs[column] = Xs[column].astype("int64")
                            elif mlm_dtype == "date":
                                Xs[column] = Xs[column].astype("datetime64[ns]")
                            elif mlm_dtype == "nominal":
                                Xs[column] = Xs[column].astype("category")
                            elif mlm_dtype == "ordinal":
                                Xs[column] = Xs[column].astype("category")
                    except KeyError:
                        continue

        return Xs

class KFoldSelectEncoder(BaseEstimator, TransformerMixin):
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

        # map column suffix to desired name
        if self.column_suffix == "_woe":
            self.column_suffix = "_woe_encoded"
        if self.column_suffix == "_cat_boost":
            self.column_suffix = "_catboost_encoded"
        elif self.column_suffix == "_target_encoder":
            self.column_suffix = "_target_encoded"

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
                X_train, X_valid = X.iloc[train_ix], X.iloc[valid_ix]

                for column in self.columns:
                    enc = self.encoder(cols=[column])

                    # fit on train
                    X_train[column + self.column_suffix] = enc.fit_transform(X_train[column], X_train[self.target.name])
                    X_valid[column + self.column_suffix] = enc.transform(X_valid[column])
                    X.loc[X.index[valid_ix], column + self.column_suffix] = X_valid[column + self.column_suffix].map(X_valid[column + self.column_suffix]).fillna(X_valid[column + self.column_suffix])

            # fill nan's with column mean. rarely needed - only when column is severely imbalanced
            for column in X.columns:
                if X[column].isnull().sum() > 0:
                    X[column] = X[column].fillna(X[column].mean())

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

class DualTransformer(TransformerMixin, BaseEstimator):
    """
    Documentation:
        Description:
            performs yeo-johnson transformation on all specified feature. also performs box_cox transformation. if
            the minimum value in a feature is 0, box_cox + 1 transformation is performed. if minimum value is greater
            than 0, standard box_cox transformation is performed. each transformation process automatically determines
            the optimal lambda value. these values are stored for transformation of validation data.

            note - this method adds additional columns rather than updating existing columns inplace.
        Parameters:
            yeojohnson : bool, default=True
                conditional controls whether yeo-johnson transformation is applied to input data.
            boxcox : bool, default=True
                conditional controls whether box_cox transformation is applied to input data.
        Returns:
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
            yj = PowerTransformer(method="yeo-johnson")
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
    Documentation:
        Description:
            displays Pandas DataFrame summarizing the skew of each number variable. also summarizes
            the percent of a column that has a value of zero.
        Parameters:
            data : Pandas DataFrame, default=None
                Pandas DataFrame containing independent variables. if left as none,
                the feature dataset provided to machine during instantiation is used.
            columns : list of strings, default=None
                list containing string names of columns. if left as none, the value associated
                with self.data.mlm_dtypes["continuous"] will be used as the column list.
    """
    # use data/mlm_dtypes["continuous"] columns provided during instantiation if left unspecified
    if data is None:
        data = self.data
    if columns is None:
        columns = self.data.mlm_dtypes["continuous"]

    skewness = (
        data[columns]
        .apply(lambda X: stats.skew(X.dropna()))
        .sort_values(ascending=False)
    )
    skewness = pd.DataFrame({"Skew": skewness})

    # add column describing percent of values that are zero
    skewness["Percent zero"] = np.nan
    for col in columns:

        try:
            skewness.loc[col]["Percent zero"] = data[data[col] == 0][
                col
            ].value_counts() / len(data)
        except ValueError:
            skewness.loc[col]["Percent zero"] = 0.0
    skewness = skewness.sort_values(["Skew"])

    return skewness

def missing_summary(self, data=None):
    """
    Documentation:
        Description:
            displays Pandas DataFrame summarizing the missingness of each variable.
        Parameters:
            data : Pandas DataFrame, default=None
                Pandas DataFrame containing independent variables. if left as none,
                the feature dataset provided to machine during instantiation is used.
    """
    # use data/target provided during instantiation if left unspecified
    if data is None:
        data = self.data

    # calcule missing data statistics
    total_missing = data.isnull().sum()
    percent_missing = data.isnull().sum() / len(data) * 100
    percent_missing = pd.DataFrame(
        {"Total missing": total_missing, "Percent missing": percent_missing}
    )
    percent_missing = percent_missing[
        ~(percent_missing["Percent missing"].isna())
        & (percent_missing["Percent missing"] > 0)
    ].sort_values(["Percent missing"], ascending=False)

    return percent_missing

def unique_category_levels(self, data=None):
    """
    Documentation:
        Description:
            displays Pandas DataFrame summarizing the skew of each number variable. also summarizes
            the percent of a column that has a value of zero.
        Parameters:
            data : Pandas DataFrame, default=None
                Pandas DataFrame containing independent variables. if left as none,
                the feature dataset provided to machine during instantiation is used.
    """

    # use data/mlm_dtypes["continuous"] columns provided during instantiation if left unspecified
    if data is None:
        data = self.data

    # print unique values in each category columns
    for column in data.mlm_dtypes["category"]:
        print(column, "\t", np.unique(data[column]))

def compare_train_valid_levels(self, train_data, validation_data):
    """
    Documentation:
        Description:
            displays Pandas DataFrame summarizing the skew of each number variable. also summarizes
            the percent of a column that has a value of zero.
        Parameters:
            train_data : Pandas DataFrame
                Pandas DataFrame containing training data.
            validation_data : Pandas DataFrame
                Pandas DataFrame containing validation data.
    """
    counter = 0
    for col in train_data.mlm_dtypes["category"]:
        train_values = train_data[col].unique()
        valid_values = validation_data[col].unique()

        train_diff = set(train_values) - set(valid_values)
        valid_diff = set(valid_values) - set(train_values)

        if len(train_diff) > 0 or len(valid_diff) > 0:
            print("\n\n*** " + col)
            print("Value present in training data, not in validation data")
            print(train_diff)
            print("Value present in validation data, not in training data")
            print(valid_diff)
            counter+=1

    # if all levels present in both datasets
    if counter == 0:
        print("All levels in all category columns present in both datasets.")

def missing_col_compare(self, train_data, validation_data):
    """
    Documentation:
        Description:
            compares the columns that contain missing data in the training dataset
            to the columns that contain missing data in the validation dataset. prints
            a summary of the disjunction.
        Parameters:
            train_data : Pandas DataFrame
                Pandas DataFrame containing training data.
            validation_data : Pandas DataFrame
                Pandas DataFrame containing validation data.
    """
    train_missing = train_data.isnull().sum()
    train_missing = train_missing[train_missing > 0].index

    validation_missing = validation_data.isnull().sum()
    validation_missing = validation_missing[validation_missing > 0].index

    print("Feature has missing values in validation data, not training data.")
    print(set(validation_missing) - set(train_missing))
    print("")
    print("Feature has missing values in training data, not validation data.")
    print(set(train_missing) - set(validation_missing))


class PreserveMetaData(pd.DataFrame):

    _metadata = ["mlm_dtypes"]

    @property
    def _constructor(self):
        return PreserveMetaData