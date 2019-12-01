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
    documentation:
        description:
            impute numerical columns with certain value, as specified by the strategy parameter. also
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
            train : boolean, default=True
                tells class whether we are imputing training data or unseen data.
            train_value : dict, default =None
                only used when train=False. value is a dictionary containing 'feature : value' pairs.
                dictionary is retrieved from training data pipeline using named steps. the attribute is
                called train_value_.
        returns:
            x : array
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

    def fit(self, x, y=None):
        if self.strategy == "mean":
            self.train_value = (
                x[x[self.null_col].notnull()]
                .groupby(self.context_col)
                .mean()[self.null_col]
            )
        elif self.strategy == "median":
            self.train_value = (
                x[x[self.null_col].notnull()]
                .groupby(self.context_col)
                .median()[self.null_col]
            )
        elif self.strategy == "most_frequent":
            self.train_value = (
                x[x[self.null_col].notnull()]
                .groupby(self.context_col)[self.null_col]
                .agg(lambda x: x.value_counts().index[0])
            )

        self.train_value = self.train_value.reset_index()
        return self

    def transform(self, x):
        # impute missing values based on train_value
        if isinstance(self.context_col, str):
            x[self.null_col] = np.where(
                x[self.null_col].isnull(),
                x[self.context_col].map(
                    self.train_value.set_index(self.context_col)[self.null_col]
                ),
                x[self.null_col],
            )
        return x[self.null_col]


class DataFrameSelector(base.BaseEstimator, base.TransformerMixin):
    """
    documentation:
        description:
            select a susbset set of features of a pandas DataFrame as part of a
            pipeline.
        parameters:
            attribute_names : list
                list of features to select from DataFrame.
    """

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.attribute_names]


class PlayWithPandas(base.TransformerMixin, base.BaseEstimator):
    """
    documentation:
        description:
            wrapper that ensures sklearn transformers will play nicely with pandas.
            pipelines will output a pandas DataFrame with column names and original
            index values rather than a two dimensional numpy array.
        parameters:
            attribute_names : list
                list of features to select from DataFrame to be passed into the
                wrapped transformer..
    """

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, x, y=None):
        # capture original column names and dtypes
        self.original_columns = x.columns
        self.dtypes = x[self.original_columns].dtypes

        # create object containing fitted class
        self.est = self.transformer.fit(x)

        # if the class is a OneHotEncoder instance
        if isinstance(self.est, preprocessing._encoders.OneHotEncoder):
            names = self.est.get_feature_names()

            prefixes = list(map(lambda x: str.replace(x, "x", ""), names))
            prefixes = sorted(list(set([x.split("_")[0] for x in prefixes])), key=int)
            prefixes = ["x" + x + "_" for x in prefixes]

            # repalce "x0_, x1_..." prefix with original column anme
            # prefixes = sorted(list(set([x.split("_")[0] for x in names])))
            for a, b in zip(self.original_columns, prefixes):
                names = list(map(lambda x: str.replace(x, b, a + "_"), names))
            self.original_columns = names

        # if the class is a polynomial_features instance
        elif isinstance(self.est, preprocessing.data.PolynomialFeatures):

            ### replace feature code names with actual feature names
            # capture object's feature code names
            names = self.est.get_feature_names()

            # remove "x" prefix from features code names
            feature_code_names = list(map(lambda x: str.replace(x, "x", ""), names))

            # reduce feature_code_names down to [0,1,2,3...] by ignoring "^2" and sort as integers
            feature_code_names = sorted(
                list(
                    set([x.replace("^2", "").split(" ")[0] for x in feature_code_names])
                ),
                key=int,
            )

            # add "x" as a prefix and an "_" as a suffix to the
            feature_code_names = ["x" + x + "_" for x in feature_code_names]

            ### build actual feature names
            # replace " " with "_ " and add an additional "_" to the full string yield names such as "x0_ x1_"
            feature_actual_names = [x.replace(" ", "_ ") + "_" for x in names]

            # change strings such as "x02_" to yield names such as "x0_2"
            feature_actual_names = [
                x.replace("^2_", "_^2") for x in feature_actual_names
            ]

            # change strings such as "x0_ x1_" to yield names such as "x0_*x1_"
            feature_actual_names = [x.replace(" ", "*") for x in feature_actual_names]

            # replace code names with actual column names
            for a, b in zip(self.original_columns, feature_code_names):
                feature_actual_names = list(
                    map(lambda x: str.replace(x, b, a), feature_actual_names)
                )

            self.original_columns = feature_actual_names

        # if the class is a KBinsDiscretizer instance
        elif isinstance(self.est, preprocessing._discretization.KBinsDiscretizer):
            names = []
            bins = self.est.n_bins
            for col in self.original_columns:
                names.append(col + "_" + str(bins) + "_bins")

            self.original_columns = names

        # if the class is a category_encoders CountEncoder instance
        elif isinstance(self.est, ce.count.CountEncoder):
            names = []

            for col in self.original_columns:
                names.append(col + "_count")

            self.original_columns = names

        # if the class is a category_encoders BinaryEncoder instance
        elif isinstance(self.est, ce.binary.BinaryEncoder):
            names = []
            self.original_columns = self.est.get_feature_names()

            for col in self.original_columns:
                names.append(col + "_binarized")

            self.original_columns = names

        # if the class is a category_encoders BinaryEncoder instance
        elif isinstance(self.est, preprocessing.data.QuantileTransformer):
            names = []

            for col in self.original_columns:
                names.append(col + "_quantile_" + self.est.output_distribution)

            self.original_columns = names
        return self

    def transform(self, x, y=None, copy=None):
        self.index = x.index

        x = self.est.transform(x)
        if sparse.issparse(x):
            x = x.todense()

        # create new pandas DataFrame
        x = pd.DataFrame(x, columns=self.original_columns, index=self.index)

        return x


class KFfoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):
    def __init__(
        self, target, cv, n_bins=5, drop_bin_cols=True, drop_original_cols=False
    ):
        self.target = target
        self.cv = cv
        self.transform_train = False
        self.n_bins = n_bins
        self.drop_bin_cols = drop_bin_cols
        self.drop_original_cols = drop_original_cols

    def fit(self, x, y=None):
        self.cols = x.columns
        self.stats = {}
        self.binners = {}

        # indentify any numeric columns in input dataset
        self.numeric_cols = x.select_dtypes(include=np.number).columns.tolist()
        return self

    def transform(self, x, y=None):
        if not self.transform_train:
            # combine input columns and target
            x = x.merge(self.target, left_index=True, right_index=True)

            # add empty columns to input dataset and set as numeric
            for col in self.cols:
                x[col + "_" + "target_encoded"] = np.nan
                x[col + "_" + "target_encoded"] = pd.to_numeric(
                    x[col + "_" + "target_encoded"]
                )

            # if column is numeric, then bin column prior to target encoding
            for col in self.numeric_cols:
                if isinstance(self.n_bins, dict):
                    binner = preprocessing.KBinsDiscretizer(
                        n_bins=self.n_bins[col], encode="ordinal"
                    )
                    # x[col] = binner.fit_transform(x[[col]])
                    x[
                        "{}_{}_bins".format(col, self.n_bins[col])
                    ] = binner.fit_transform(x[[col]])

                    # store binner transformer for each column
                    self.binners[col] = binner

                else:
                    binner = preprocessing.KBinsDiscretizer(
                        n_bins=self.n_bins, encode="ordinal"
                    )
                    # x[col] = binner.fit_transform(x[[col]])
                    x["{}_{}_bins".format(col, self.n_bins)] = binner.fit_transform(
                        x[[col]]
                    )

                    # store binner transformer for each column
                    self.binners[col] = binner

            # iterate through cv indices
            for train_ix, valid_ix in self.cv.split(x):
                x_train, x_valid = x.iloc[train_ix], x.iloc[valid_ix]

                # update rows with out of fold averages
                for col in self.cols:
                    if col in self.numeric_cols:
                        if isinstance(self.n_bins, dict):
                            x.loc[
                                x.index[valid_ix], col + "_" + "target_encoded"
                            ] = x_valid["{}_{}_bins".format(col, self.n_bins[col])].map(
                                x_train.groupby(
                                    "{}_{}_bins".format(col, self.n_bins[col])
                                )[self.target.name].mean()
                            )
                        else:
                            x.loc[
                                x.index[valid_ix], col + "_" + "target_encoded"
                            ] = x_valid["{}_{}_bins".format(col, self.n_bins)].map(
                                x_train.groupby("{}_{}_bins".format(col, self.n_bins))[
                                    self.target.name
                                ].mean()
                            )
                    else:
                        x.loc[
                            x.index[valid_ix], col + "_" + "target_encoded"
                        ] = x_valid[col].map(
                            x_train.groupby(col)[self.target.name].mean()
                        )

            # ensure numeric data type
            for col in self.cols:
                x[col + "_" + "target_encoded"] = pd.to_numeric(
                    x[col + "_" + "target_encoded"]
                )

            # collect average values for transformation of unseen data
            for col in self.cols:
                if col in self.numeric_cols:
                    if isinstance(self.n_bins, dict):
                        self.stats[col] = x.groupby(
                            "{}_{}_bins".format(col, self.n_bins[col])
                        )["{}_target_encoded".format(col)].mean()
                    else:
                        self.stats[col] = x.groupby(
                            "{}_{}_bins".format(col, self.n_bins)
                        )["{}_target_encoded".format(col)].mean()
                else:
                    self.stats[col] = x.groupby(col)[
                        "{}_target_encoded".format(col)
                    ].mean()

            # flip transform_train switch to indicate that fitting has occurred
            # ensure future transform calls will go to the else branch of the conditional
            self.transform_train = True

            # drop target column
            x = x.drop(self.target.name, axis=1)

            # conditionally drop original and/or binned columns
            if self.drop_original_cols:
                x = x.drop(self.cols, axis=1)
            if self.drop_bin_cols:
                x = x.drop(x.columns[x.columns.str.contains("_bins")], axis=1)
        else:
            # ieterate through all columns in the summary stats dict
            for col in self.stats.keys():

                # add shell column and update by mapping encodings to categorical values or bins
                x["{}_target_encoded".format(col)] = np.nan

                # perform binning on numeric columns
                if col in self.numeric_cols:
                    binner = self.binners[col]

                    if isinstance(self.n_bins, dict):

                        x[
                            col + "_" + str(self.n_bins[col]) + "_bins"
                        ] = binner.transform(x[[col]])

                        x["{}_target_encoded".format(col)] = np.where(
                            x["{}_target_encoded".format(col)].isnull(),
                            x["{}_{}_bins".format(col, self.n_bins[col])].map(
                                self.stats[col]
                            ),
                            x["{}_target_encoded".format(col)],
                        )
                    else:
                        x[col + "_" + str(self.n_bins) + "_bins"] = binner.transform(
                            x[[col]]
                        )

                        x["{}_target_encoded".format(col)] = np.where(
                            x["{}_target_encoded".format(col)].isnull(),
                            x["{}_{}_bins".format(col, self.n_bins)].map(
                                self.stats[col]
                            ),
                            x["{}_target_encoded".format(col)],
                        )
                else:
                    x["{}_target_encoded".format(col)] = np.where(
                        x["{}_target_encoded".format(col)].isnull(),
                        x[col].map(self.stats[col]),
                        x["{}_target_encoded".format(col)],
                    )

            # conditionally drop original and/or binned columns
            if self.drop_original_cols:
                x = x.drop(self.cols, axis=1)
            if self.drop_bin_cols:
                x = x.drop(x.columns[x.columns.str.contains("_bins")], axis=1)
        return x


class PandasFeatureUnion(pipeline.FeatureUnion):
    """
    documentation:
        description:
            modified version of sklearn's FeatureUnion class that outputs a pandas DataFrame rather
            than a two dimensional array. credit to x for this idea. please take a look at the creator's
            original article detailing this function:

            https://zablo.net/blog/post/pandas_dataframe_in_scikit_learn_FeatureUnion/index.html
    """

    def fit_transform(self, x, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(pipeline._fit_transform_one)(
                transformer=trans, x=x, y=y, weight=weight, **fit_params
            )
            for name, trans, weight in self._iter()
        )

        if not result:
            # all transformers are none
            return np.zeros((x.shape[0], 0))
        xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in xs):
            xs = sparse.hstack(xs).tocsr()
        else:
            xs = self.merge_dataframes_by_column(xs)
        return xs

    def merge_dataframes_by_column(self, xs):
        return pd.concat(xs, axis="columns", copy=False)

    def transform(self, x):
        xs = Parallel(n_jobs=self.n_jobs)(
            delayed(pipeline._transform_one)(
                transformer=trans, x=x, y=None, weight=weight
            )
            for name, trans, weight in self._iter()
        )
        if not xs:
            # all transformers are none
            return np.zeros((x.shape[0], 0))
        if any(sparse.issparse(f) for f in xs):
            xs = sparse.hstack(xs).tocsr()
        else:
            xs = self.merge_dataframes_by_column(xs)
        return xs


class UnprocessedColumnAdder(base.TransformerMixin, base.BaseEstimator):
    """
    documentation:
        description:
            add the columns not processed by other transformers in pipeline to the final
            output dataset.
        parameters:
            pipes : sklearn FeatureUnion pipeline object
                FeatureUnion pipeline object with all column_specific transformations.
    """

    def __init__(self, pipes):
        self.pipes = pipes
        self.processed_columns = []

    def fit(self, x, y=None):

        # for each transformer in the input pipe
        for pipe_trans in self.pipes.transformer_list:
            pipe_cols = []

            # for each step in each transformer
            for step in pipe_trans[1].steps:
                # add all columns specified in DataFrameSelector objects
                if isinstance(step[1], DataFrameSelector):
                    pipe_cols.append(step[1].attribute_names)
                    pipe_cols = list(set(itertools.chain(*pipe_cols)))
                # remove all columns specified in ContextImputer objects
                elif isinstance(step[1], ContextImputer):
                    remove_cols = step[1].context_col
                    if isinstance(remove_cols, str):
                        pipe_cols = [i for i in pipe_cols if i not in [remove_cols]]

            # collect columns in processed_columns attribute
            self.processed_columns.append(pipe_cols)

        # flatten processed_columns
        self.processed_columns = list(set(itertools.chain(*self.processed_columns)))

        # compare columns of input dataset with processed_columns to identify unprocessed columns
        self.unprocessed_columns = list(set(x.columns) - set(self.processed_columns))
        return self

    def transform(self, x):
        return x[self.unprocessed_columns]


class DualTransformer(base.TransformerMixin, base.BaseEstimator):
    """
    documentation:
        description:
            performs yeo_johnson transformation on all specified feature. also performs box_cox transformation. if
            the minimum value in a feature is 0, box_cox + 1 transformation is performed. if minimum value is greater
            than 0, standard box_cox transformation is performed. each transformation process automatically determines
            the optimal lambda value. these values are stored for transformation of validation data.

            note - this method adds additional columns rather than updating existing columns inplace.
        parameters:
            yeojohnson : boolean, default=True
                conditional controls whether yeo_johnson transformation is applied to input data.
            boxcox : boolean, default=True
                conditional controls whether box_cox transformation is applied to input data.
        returns:
            x : array
                yeo_johnson, and potentially box_cox (or box_cox + 1) transformed input data.
    """

    def __init__(self, yeojohnson=True, boxcox=True):
        self.yeojohnson = yeojohnson
        self.boxcox = boxcox

    def fit(self, x, y=None):
        self.original_columns = x.columns

        ## yeo_johnson
        if self.yeojohnson:

            self.yj_lambdas_dict_ = {}

            # collect lambas using sklearn implementation
            yj = preprocessing.PowerTransformer(method="yeo_johnson")
            yj_lambdas = yj.fit(x).lambdas_

            # cycle through columns, add transformed columns, store lambdas
            for ix, col in enumerate(x.columns):
                self.yj_lambdas_dict_[col] = yj_lambdas[ix]

        ## box_cox
        if self.boxcox:

            self.bc_p1_lambdas_dict_ = {}
            self.bc_lambdas_dict_ = {}

            for col in x.columns:
                # if minimum feature value is 0, do box_cox + 1
                if np.min(x[col]) == 0:
                    _, lmbda = stats.boxcox(x[col].values + 1, lmbda=None)
                    self.bc_p1_lambdas_dict_[col] = lmbda
                # otherwise do standard box_cox
                elif np.min(x[col]) > 0:
                    _, lmbda = stats.boxcox(x[col].values, lmbda=None)
                    self.bc_lambdas_dict_[col] = lmbda
        return self

    def transform(self, x):
        # yeo_johnson
        if self.yeojohnson:
            for col in self.yj_lambdas_dict_.keys():
                x[col + "_yj"] = stats.yeojohnson(
                    x[col].values, lmbda=self.yj_lambdas_dict_[col]
                )

        # box_cox
        if self.boxcox:
            for col in self.bc_p1_lambdas_dict_.keys():
                x[col + "_bc"] = stats.boxcox(
                    x[col].values + 1, lmbda=self.bc_p1_lambdas_dict_[col]
                )

            for col in self.bc_lambdas_dict_.keys():
                x[col + "_bc"] = stats.boxcox(
                    x[col].values, lmbda=self.bc_lambdas_dict_[col]
                )
        return x
        # return x.drop(self.original_columns, axis=1)


def skew_summary(self, data=None, columns=None):
    """
    documentation:
        description:
            displays pandas DataFrame summarizing the skew of each numeric variable. also summarizes
            the percent of a column that has a value of zero.
        parameters:
            data : pandas DataFrame, default =None
                pandas DataFrame containing independent variables. if left as none,
                the feature dataset provided to machine during instantiation is used.
            columns : list of strings, default =None
                list containing string names of columns. if left as none, the value associated
                with sel.feature_type["numeric"] will be used as the column list.
    """
    # use data/feature_type["numeric"] columns provided during instantiation if left unspecified
    if data is None:
        data = self.data
    if columns is None:
        columns = self.feature_type["numeric"]

    skewness = (
        data[columns]
        .apply(lambda x: stats.skew(x.dropna()))
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
