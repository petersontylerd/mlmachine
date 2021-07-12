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
from joblib import Parallel, delayed

from category_encoders import WOEEncoder, TargetEncoder, CatBoostEncoder, BinaryEncoder, CountEncoder

import itertools
import collections
import copy


class GroupbyImputer(TransformerMixin, BaseEstimator):
    """
    Documentation:

        ---
        Description:
            Impute numeric columns, as specified by the strategy parameter. GroupbyImputer utilizes one or
            more additional context columns as a groupby value to add more subtlety to fill_value identification.
            Imputes training data features, and stores impute values to be used on validation and unseen data.

        ---
        Parameters:
            null_column : list
                Column with nulls to impute.
            groupby_column : list
                List of one or most columns to groupby to add context to null column.
            strategy : str, default='mean'
                Imputing stategy. Accepts values 'mean', 'median' and 'most_frequent'.
            train : bool, default=True
                Tells class whether we are imputing training data or unseen data.
            train_value : dict, default=None
                Only used when train=False. Value is a dictionary containing 'feature : value' pairs.
                Dictionary is retrieved from training data pipeline using named steps. The attribute is
                called train_value_.

        ---
        Returns:
            X : array
                Dataset where each column with missing values has been imputed with a value learned from a particular
                strategy while also consider select columns as a group by variable.
    """

    def __init__(self, null_column, groupby_column, strategy="mean", train=True, train_value=None):
        self.null_column = null_column
        self.groupby_column = groupby_column
        self.strategy = strategy
        self.train = train
        self.train_value = train_value

    def fit(self, X, y=None):

        # if imputation strategy is set to "mean"
        if self.strategy == "mean":

            # grouping by groupby_column, find mean of null_column
            self.train_value = (
                X[X[self.null_column].notnull()]
                .groupby(self.groupby_column)
                .mean()[self.null_column]
            )
            # calculate overall mean of null_column
            self.overall = X[X[self.null_column].notnull()][self.null_column].mean()

        # if imputation strategy is set to "median"
        elif self.strategy == "median":
            # grouping by groupby_column, find median of null_column
            self.train_value = (
                X[X[self.null_column].notnull()]
                .groupby(self.groupby_column)
                .median()[self.null_column]
            )
            # calculate overall median of null_column
            self.overall = X[X[self.null_column].notnull()][self.null_column].median()

        # if imputation strategy is set to "most_frequent"
        elif self.strategy == "most_frequent":
            # grouping by groupby_column, find mode of null_column
            self.train_value = (
                X[X[self.null_column].notnull()]
                .groupby(self.groupby_column)[self.null_column]
                .agg(lambda X: X.value_counts().index[0])
            )
            # calculate overall mode of null_column
            self.overall = X[X[self.null_column].notnull()][self.null_column].mode()[0]

        self.train_value = self.train_value.reset_index()

        return self

    def transform(self, X):
        # impute missing values based on train_value
        if isinstance(self.groupby_column, str):

            # impute nulls with corresponding value
            X[self.null_column] = np.where(
                X[self.null_column].isnull(),
                X[self.groupby_column].map(
                    self.train_value.set_index(self.groupby_column)[self.null_column]
                ),
                X[self.null_column],
            )

            # impute any remainig nulls with overall value
            X[self.null_column] = X[self.null_column].fillna(value=self.overall)

        return X[self.null_column]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Documentation:

        ---
        Description:
            Select a susbset set of features from Pandas DataFrame as part of a
            pipeline. Capable of select and/or deselecting columns by name, Pandas
            dtype and mlm dtype.

            Note - if there is a logical conflict between include and exclude
            parameters, the class will first prioritize the column parameters
            over the dtype parameters in order to support subsetting. If the
            logic cannot be resolved by this rule alone, exclusion parameters
            will be prioritized over inclusion parameters.

        ---
        Parameters:
            include_columns : list
                List of features to select from Pandas DataFrame.
            include_pd_dtypes : list
                List of strings describing pandas dtypes to select.
            include_mlm_dtypes : list
                List of strings describing mlmachine dtypes to select.
            exclude_columns : list
                List of features to exclude from Pandas DataFrame.
            exclude_pd_dtypes : list
                List of strings describing pandas dtypes to exclude.
            exclude_mlm_dtypes : list
                List of strings describing mlmachine dtypes to exclude.
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

        # preserve mlm_dtypes attribute on Pandas DataFrame, if it exists
        try:
            self.meta_mlm_dtypes = X.mlm_dtypes
            self.no_meta_mlm_dtypes = False
        except AttributeError:
            if self.include_mlm_dtypes is not None or self.exclude_mlm_dtypes is not None:
                raise AttributeError ("Attempting to filter using mlm dtypes, but mlm_dtypes object is not associated with the input Pandas DataFrame.")
            else:
                pass

        # capture all current columns in input Pandas DataFrame
        self.all_columns = X.columns.tolist()

        # empty list to capture selected columns
        self.selected_columns = []

        # empty list to capture excluded columns
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

        # select columns by mlm dtype
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

        # deselect columns by mlm dtype
        if self.exclude_mlm_dtypes is not None:
            for dtype in self.exclude_mlm_dtypes:
                self.remove_columns.extend(X.mlm_dtypes[dtype])

        # flatten list and remove duplicates
        self.remove_columns = list(set(self.remove_columns))

        ## reconcile selections and removals
        # if the only input is to exclude, remove remove_columns from all_columns
        if len(self.remove_columns) > 0 and len(self.selected_columns) == 0:
            # set final_column to all_columns minus the columns in remove_columns
            self.final_columns = list(
                set(self.all_columns).difference(self.remove_columns)
            )

        # if the only input is to include, keep only the select_columns
        elif len(self.selected_columns) > 0 and len(self.remove_columns) == 0:
            # set final_column to selected_columns
            self.final_columns = self.selected_columns

        # if columns are selected by name and columns are excluded by name
        elif (
            self.include_columns is not None
            and (self.include_pd_dtypes is None and self.include_mlm_dtypes is None)
            and self.exclude_columns is not None
            and (self.exclude_pd_dtypes is None and self.exclude_mlm_dtypes is None)
        ):
            # set final_column to selected_columns and remove all columns in remove_columns
            self.final_columns = self.selected_columns
            self.final_columns = [column for column in self.final_columns if column not in self.remove_columns]

        # if columns are selected by name and columns are excluded by pandas dtype or mlm dtype
        elif (
            self.include_columns is not None
            and (self.include_pd_dtypes is None and self.include_mlm_dtypes is None)
            and self.exclude_columns is None
            and (self.exclude_pd_dtypes is not None or self.exclude_mlm_dtypes is not None)
        ):
            # set final_column to selected_columns
            self.final_columns = self.selected_columns

        # if columns are excluded by name and columns are selected by pandas dtype or mlm dtype
        elif (
            self.include_columns is None
            and (self.include_pd_dtypes is not None or self.include_mlm_dtypes is not None)
            and self.exclude_columns is not None
            and (self.exclude_pd_dtypes is None and self.exclude_mlm_dtypes is None)
        ):
            # set final_column to selected_columns and remove all columns in remove_columns
            self.final_columns = self.selected_columns
            self.final_columns = [column for column in self.final_columns if column not in self.remove_columns]

        #
        # if columns are selected by pandas dtype or mlm dtype, and  columns are excluded by pandas dtype or mlm dtype
        elif (
            self.include_columns is None
            and (self.include_pd_dtypes is not None or self.include_mlm_dtypes is not None)
            and self.exclude_columns is None
            and (self.exclude_pd_dtypes is not None or self.exclude_mlm_dtypes is not None)
        ):
            # set final_column to selected_columns and remove all columns in remove_columns
            self.final_columns = self.selected_columns
            self.final_columns = [column for column in self.final_columns if column not in self.remove_columns]

        # if columns are selected by pandas dtype or mlm dtype
        elif (
            self.include_columns is not None
            and (self.include_pd_dtypes is not None or self.include_mlm_dtypes is not None)
            and self.exclude_columns is not None
            and (self.exclude_pd_dtypes is None and self.exclude_mlm_dtypes is None)
        ):
            # set final_column to selected_columns and remove all columns in remove_columns
            self.final_columns = self.selected_columns
            self.final_columns = [column for column in self.final_columns if column not in self.remove_columns]

        # if columns are selected by name, and selected by pandas dtype or mlm dtype, and columns are excluded
        # by pandas dtype or mlm dtype
        elif (
            self.include_columns is not None
            and (self.include_pd_dtypes is not None or self.include_mlm_dtypes is not None)
            and self.exclude_columns is None
            and (self.exclude_pd_dtypes is not None or self.exclude_mlm_dtypes is not None)
        ):
            # set final_column to selected_columns
            self.final_columns = self.selected_columns

        # if columns are excluded by name, and selected by pandas dtype or mlm dtype, and columns are excluded
        # by pandas dtype or mlm dtype
        elif (
            self.include_columns is None
            and (self.include_pd_dtypes is not None or self.include_mlm_dtypes is not None)
            and self.exclude_columns is not None
            and (self.exclude_pd_dtypes is not None or self.exclude_mlm_dtypes is not None)
        ):
            # set final_column to selected_columns and remove all columns in remove_columns
            self.final_columns = self.selected_columns
            self.final_columns = [column for column in self.final_columns if column not in self.remove_columns]

        # if columns are selected by name and excluded by name, and excluded by pandas dtype or mlm dtype
        elif (
            self.include_columns is not None
            and (self.include_pd_dtypes is None and self.include_mlm_dtypes is None)
            and self.exclude_columns is not None
            and (self.exclude_pd_dtypes is not None or self.exclude_mlm_dtypes is not None)
        ):
            # set final_column to selected_columns
            self.final_columns = self.selected_columns

        # if columns are selected by name and excluded by name, and selected by pandas dtype or mlm dtype, and
        # excluded by pandas dtype or mlm dtype
        elif (
            self.include_columns is not None
            and (self.include_pd_dtypes is not None or self.include_mlm_dtypes is not None)
            and self.exclude_columns is not None
            and (self.exclude_pd_dtypes is not None or self.exclude_mlm_dtypes is not None)
        ):
            # set final_column to selected_columns
            self.final_columns = self.selected_columns

        # if all parameters are left to default to None
        elif (
            self.include_columns is None
            and self.include_pd_dtypes is None
            and self.include_mlm_dtypes is None
            and self.exclude_columns is None
            and self.exclude_pd_dtypes is None
            and self.exclude_mlm_dtypes is None
        ):

            # set final_column to all_columns
            self.final_columns = self.all_columns

        return self

    def transform(self, X):
        return X[self.final_columns]

class PandasTransformer(TransformerMixin, BaseEstimator):
    """
    Documentation:

        ---
        Description:
            Wrapper that ensures sklearn transformers will play nicely with Pandas.
            Pipelines will output a Pandas DataFrame with column names and original
            index values rather than a two dimensional numpy array.

        ---
        Parameters
            transformer : sklearn transformer
                Transformer to apply to dataset
    """
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):

        # capture original column names and dtypes
        self.original_columns = X.columns
        self.dtypes = X[self.original_columns].dtypes

        # create object containing fitted transformer
        self.est = self.transformer.fit(X)

        # if the class is a OneHotEncoder instance
        if isinstance(self.est, _encoders.OneHotEncoder):

            # capture feature names learned by OneHotEncoder
            names = self.est.get_feature_names()

            # transform names into readable column names
            prefixes = list(map(lambda x:str.replace(x, "x", ""), names))
            prefixes = sorted(list(set([x.split("_")[0] for x in prefixes])), key=int)
            prefixes = ["x" + x + "_" for x in prefixes]

            # add "_{category name}" suffix to all column names
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

            # capture number of bins learned by KBinsDiscretizer
            bins = self.est.n_bins

            # add "_binned_{# of bins}" suffix to all column names
            for col in self.original_columns:
                names.append(col + "_binned_" + str(bins))

            self.original_columns = names

        # if the class is a OrdinalEncoder instance
        elif isinstance(self.est, OrdinalEncoder):
            names = []

            # add "_ordinal_encoded" suffix to all column names
            for col in self.original_columns:
                names.append(col + "_ordinal_encoded")

            self.original_columns = names

        # if the class is a category_encoders CountEncoder instance
        elif isinstance(self.est, CountEncoder):
            names = []

            # add "_count_encoded" suffix to all column names
            for col in self.original_columns:
                names.append(col + "_count_encoded")

            self.original_columns = names

        # if the class is a category_encoders BinaryEncoder instance
        elif isinstance(self.est, BinaryEncoder):
            names = []
            self.original_columns = self.est.get_feature_names()

            # add "_binary_encoded" suffix to all column names
            for col in self.original_columns:
                names.append(col + "_binary_encoded")

            self.original_columns = names

        # if the class is a category_encoders BinaryEncoder instance
        elif isinstance(self.est, QuantileTransformer):
            names = []

            # add "_quantile_{distribution type}" suffix to all column names
            for col in self.original_columns:
                names.append(col + "_quantile_" + self.est.output_distribution)

            self.original_columns = names
        return self

    def transform(self, X, y=None, copy=None):
        self.index = X.index

        # peform apply learned transformation to input dataset
        X = self.est.transform(X)

        # if data is sparse, convert to dense
        if sparse.issparse(X):
            X = X.todense()

        # create new Pandas DataFrame with new column names
        X = pd.DataFrame(X, columns=self.original_columns, index=self.index)
        return X

class PandasFeatureUnion(FeatureUnion):
    """
    Documentation:

        ---
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

class KFoldEncoder(BaseEstimator, TransformerMixin):
    """
    Documentation:

        ---
        Description:
            Perform KFold encoding using the provided cross-validation object and encoder. KFold
            encoding prevents leakage when using encoders that incorporate the target data. KFoldEncoder
            applies a procedure where the encoder fits to the training data and then applies the encoding
            on the held out dataset. This process repeats for each fold, per the setup of the cross-validation
            object, until all observations have been encoded in this manner.

        ---
        Parameters
            target : Pandas Series
                Pandas Series containing target values
            cv : sklearn cross-validation object
                Object to use when applying cross-validation to dataset
            encoder : encoder class
                Class to use when encoding features
            transform_train : boolean, default=False
                Controls whether KFoldEncoder's transform method runs in training mode or impute mode.
                Defaults to False so that when KFoldEncoder is first utilized, it runs in training mode
                so that it learns the encoding values. Once these values are learned, transform_train is
                automatically set to True so that subsequent calls of the transform method apply the
                previously learned encoding values.
    """
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

        ---
        Description:
            Performs Yeo-Johnson transformation on all specified features. Also performs Box-Cox transformation.
            If the minimum value is greater than 0, standard Box-Cox transformation is performed. If the minimum
            value in a feature is 0, Box-Cox + 1 transformation is performed. If the minimum value in a feature is
            less than 0, Box-Cox + 1 + absolute value of the minimum value is performed. Each transformation
            process automatically determines the optimal lambda value. These values are stored for transformation
            of validation data.

            This method adds additional columns rather than updating existing columns in-place.

        ---
        Parameters:
            yeojohnson : bool, default=True
                Conditional controlling whether yeo-johnson transformation is applied to input data.
            boxcox : bool, default=True
                Conditional controlling whether Box-Cox transformation is applied to input data.

        --
        Returns:
            X : array
                Yeo-Johnson and Box-Coxtransformed input data.
    """

    def __init__(self, yeojohnson=True, boxcox=True):
        self.yeojohnson = yeojohnson
        self.boxcox = boxcox

    def fit(self, X, y=None):

        # capture original column names
        self.original_columns = X.columns

        ## Yeo-Johnson
        if self.yeojohnson:

            # empty dicitonary for capturing feature/lambda pairs
            self.yj_lambdas_dict_ = {}

            # collect lambas using sklearn PowerTransformer
            yj = PowerTransformer(method="yeo-johnson")
            yj_lambdas = yj.fit(X).lambdas_

            # cycle through columns, add transformed column name and associated lambdas
            for ix, col in enumerate(X.columns):
                self.yj_lambdas_dict_[col] = yj_lambdas[ix]

        ## Box-Cox
        if self.boxcox:

            # empty dicitonaries for capturing feature/lambda pairs
            self.bc_lambdas_dict_ = {}
            self.bc_zero_lambdas_dict_ = {}
            self.bc_neg_lambdas_dict_ = {}

            for col in X.columns:

                # if minimum feature value is 0, do Box-Cox + 1
                feature_minimum = np.min(X[col].values)
                if feature_minimum == 0:

                    # perform Box-Cox transformation and return lambda
                    _, lmbda = stats.boxcox(X[col].values + 1, lmbda=None)

                    # add transformed column name and associated lambda to dictionary
                    self.bc_zero_lambdas_dict_[col] = lmbda

                # if minimum feature value is greater than 0, do standard Box-Cox
                elif feature_minimum > 0:

                    # perform Box-Cox transformation and return lambda
                    _, lmbda = stats.boxcox(X[col].values, lmbda=None)

                    # add transformed column name and associated lambda to dictionary
                    self.bc_lambdas_dict_[col] = lmbda

                # if minimum feature value is less than 0, do Box-Cox + 1 + absolute value of minimum
                else:

                    # perform Box-Cox transformation and return lambda
                    _, lmbda = stats.boxcox(X[col].values + np.abs(feature_minimum) + 1, lmbda=None)

                    # add transformed column name and associated lambda to dictionary
                    self.bc_neg_lambdas_dict_[col] = lmbda
        return self

    def transform(self, X):
        # yeo-johnson
        if self.yeojohnson:

            # apply learned Yeo-Johnson transformation
            for col in self.yj_lambdas_dict_.keys():
                X[col + "_YeoJohnson"] = stats.yeojohnson(
                    X[col].values, lmbda=self.yj_lambdas_dict_[col]
                )

        # Box-Cox
        if self.boxcox:

            # apply learned Box-Cox transformation
            for col in self.bc_zero_lambdas_dict_.keys():
                try:
                    X[col + "_BoxCox"] = stats.boxcox(
                        X[col].values + 1, lmbda=self.bc_zero_lambdas_dict_[col]
                    )
                except ValueError:
                    X[col + "_BoxCox"] = 0.

            # apply learned Box-Cox transformation
            for col in self.bc_neg_lambdas_dict_.keys():
                try:
                    X[col + "_BoxCox"] = stats.boxcox(
                        X[col].values + np.abs(np.min(X[col].values)) + 1, lmbda=self.bc_neg_lambdas_dict_[col]
                    )
                except ValueError:
                    X[col + "_BoxCox"] = 0.

            # apply learned Box-Cox transformation
            for col in self.bc_lambdas_dict_.keys():
                try:
                    X[col + "_BoxCox"] = stats.boxcox(
                        X[col].values, lmbda=self.bc_lambdas_dict_[col]
                    )
                except ValueError:
                    X[col + "_BoxCox"] = 0.
        return X

def skew_summary(self, data=None, columns=None):
    """
    Documentation:

        ---
        Description:
            Displays Pandas DataFrame summarizing the skew of each numeric variable. Also calculates
            the percent of a column that is zero.

        ---
        Parameters:
            data : Pandas DataFrame, default=None
                Pandas DataFrame containing independent variables. If left as none,
                The feature dataset provided to Machine during instantiation is used.
            columns : list of strings, default=None
                List containing string names of columns. If left as none, the value associated
                with self.data.mlm_dtypes["continuous"] will be used as the column list.
    """
    # use data columns provided during instantiation if None
    if data is None:
        data = self.data
    # use mlm_dtypes["continuous"] columns provided during instantiation if None
    if columns is None:
        columns = self.data.mlm_dtypes["continuous"]

    # calculate skew for each column, dropping nulls and sorting on skew descending
    skewness = (
        data[columns]
        .apply(lambda X: stats.skew(X.dropna()))
        .sort_values(ascending=False)
    )

    # store results in DataFrame
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

    # fill remaining nulls with zero
    skewness["Percent zero"] = skewness["Percent zero"].fillna(0.0)

    return skewness

def missing_summary(self, data=None):
    """
    Documentation:

        ---
        Description:
            Displays Pandas DataFrame summarizing the missingness of each variable.

        ---
        Parameters:
            data : Pandas DataFrame, default=None
                Pandas DataFrame containing independent variables. If left as none,
                the feature dataset provided to Machine during instantiation is used.
    """
    # use data provided during instantiation if None
    if data is None:
        data = self.data

    # calculate missing data statistics
    total_missing = data.isnull().sum()
    percent_missing = data.isnull().sum() / len(data) * 100

    # create DataFrame using missingness data
    percent_missing = pd.DataFrame(
        {"Total missing": total_missing, "Percent missing": percent_missing}
    )

    # limit DataFrame to rows where "Percent missing" is greater than 0, sort descending
    # by "Percent missing"
    percent_missing = percent_missing[
        ~(percent_missing["Percent missing"].isna())
        & (percent_missing["Percent missing"] > 0)
    ].sort_values(["Percent missing"], ascending=False)

    return percent_missing

def unique_category_levels(self, data=None):
    """
    Documentation:

        ---
        Description:
            Print the unique values within each category column as define by mlm_dtypes["category"].

        ---
        Parameters:
            data : Pandas DataFrame, default=None
                Pandas DataFrame containing independent variables. If left as none,
                the feature dataset provided to Machine during instantiation is used.
    """

    # use data/ columns provided during instantiation if None
    if data is None:
        data = self.data

    # print unique values in each category columns
    for column in data.mlm_dtypes["category"]:
        print(column, "\t", np.unique(data[column]))

def compare_train_valid_levels(self, train_data, validation_features):
    """
    Documentation:

        ---
        Description:
            displays Pandas DataFrame summarizing the skew of each number variable. Also summarizes
            the percent of a column that has a value of zero.

        ---
        Parameters:
            train_data : Pandas DataFrame
                Pandas DataFrame containing training data.
            validation_features : Pandas DataFrame
                Pandas DataFrame containing validation data.
    """
    # creat counter to track how many features have differing amounts of unique values
    # between the input training and validation datasets
    counter = 0
    for col in train_data.mlm_dtypes["category"]:

        # capture unique feature values in both training and validation datasets
        train_values = train_data[col].unique()
        valid_values = validation_features[col].unique()

        # determine if categories exist in the training data and not the validation data
        train_diff = set(train_values) - set(valid_values)

        # determine if categories exist in the validation data and not the training data
        valid_diff = set(valid_values) - set(train_values)

        # if there is a discrepancy in the presence of category values in the training
        # and/or validation datasets, print summary information
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

def missing_column_compare(self, validation_features, train_data=None):
    """
    Documentation:

        ---
        Description:
            Compares the columns that contain missing data in the training dataset
            to the columns that contain missing data in the validation dataset. Prints
            a summary of the disjunction.

        ---
        Parameters:
            validation_features : Pandas DataFrame
                Pandas DataFrame containing validation data.
            train_data : Pandas DataFrame, default=None
                Pandas DataFrame containing training data. If no value is passed, Method
                will default to using the Machine object's data attribute.
    """
    # use data provided during instantiation if None
    if train_data is None:
        train_data = self.data

    # return DataFrame describing number of values missing within each feature in the
    # training data
    train_missing = train_data.isnull().sum()

    # limit to feature that have greater than 0 missing values
    train_missing = train_missing[train_missing > 0].index

    # return DataFrame describing number of values missing within each feature in the
    # validation data
    validation_missing = validation_features.isnull().sum()

    # limit to feature that have greater than 0 missing values
    validation_missing = validation_missing[validation_missing > 0].index

    # print bi-direction summary of which columns are missing in the training dataset
    # versus the validation dataset
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