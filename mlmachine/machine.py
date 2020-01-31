import copy
import os
import sys
import importlib
import itertools
import warnings

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

import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing


class Machine:
    """
    documentation:
        description:
            machine facilitates rapid machine learning experimentation tasks, including data
            cleaning, feature encoding, exploratory data analysis, data prepation, model building,
            model tuning and model evaluation.
    """

    # import mlmachine submodules
    from .explore.eda_missing import eda_missing_summary
    from .explore.eda_suite import (
        df_side_by_side,
        eda_cat_target_cat_feat,
        eda_cat_target_num_feat,
        eda_num_target_cat_feat,
        eda_num_target_num_feat,
    )
    from .explore.eda_transform import (
        eda_transform_box_cox,
        eda_transform_initial,
        eda_transform_log1,
    )
    from .features.preprocessing import (
        ContextImputer,
        DataFrameSelector,
        DualTransformer,
        KFoldSelectEncoder,
        PandasFeatureUnion,
        PandasPipeline,
        skew_summary,
    )
    from .features.missing import (
        missing_col_compare,
        missing_data_dropper_all,
    )
    from .features.outlier import (
        ExtendedIsoForest,
        OutlierIQR,
        outlier_summary,
        outlier_IQR,
    )
    from .features.selection import FeatureSelector
    from .features.transform import (
        custom_binner,
        equal_width_binner,
    )
    from .model.evaluate.summarize import (
        regression_results,
        regression_stats,
        top_bayes_optim_models,
    )
    from .model.evaluate.visualize import (
        classification_panel,
        regression_panel,
    )
    from .model.explain.visualize import (
        multi_shap_value_tree,
        multi_shap_viz_tree,
        shap_dependence_grid,
        shap_dependence_plot,
        shap_summary_plot,
        single_shap_value_tree,
        single_shap_viz_tree,
    )
    from .model.tune.bayesian_optim_search import (
        BasicModelBuilder,
        BayesOptimModelBuilder,
        exec_bayes_optim_search,
        model_loss_plot,
        model_param_plot,
        objective,
        sample_plot,
        unpack_bayes_optim_summary,
    )
    from .model.tune.power_grid_search import (
        PowerGridModelBuilder,
        PowerGridSearcher,
    )
    from .model.tune.stack import (
        model_stacker,
        oof_generator,
    )

    def __init__(self, data, remove_features=[], identify_as_boolean=None, identify_as_continuous=None,identify_as_count=None,
                identify_as_date=None, identify_as_nominal=None, identify_as_ordinal=None, ordinal_encodings=None,
                identify_as_string=None, target=None, target_type=None):
        """
        documentation:
            description:
                __init__ handles initial processing of main data set. returns
                data frame of independent variables, series containing dependent variable
                and a dictionary that categorizes features by data type.
            parameters:
                data : pandas DataFrame
                    input data provided as a pandas DataFrame.
                remove_features : list, default = []
                    features to be completely removed from dataset.
                identify_as_boolean : list, default=None
                    preidentified boolean features. columns given boolean dtype.
                identify_as_continuous : list, default=None
                    preidentified continuous features. columns given float64 dtype.
                identify_as_count : list, default=None
                    preidentified count features. columns given int64 dtype.
                identify_as_date : list, default=None
                    preidentified date features. columns given datetime64[ns] dtype.
                identify_as_nominal : list, default=None
                    preidentified nominal category features. columns given category dtype.
                identify_as_ordinal : list, default=None
                    preidentified ordinal category features. columns given category dtype. if
                    an ordinal_encodings dict is passed, the category column will be given the
                    specified order.
                ordinal_encodings : dict, default=None
                    dictionary where the key is the ordinal column name provided as a
                    string, and the associated value is a list containing the preferred
                    order of the values.
                identify_as_string : list, default=None
                    preidentified string features. columns given string dtype.
                target : list, default=None
                    name of column containing dependent variable.
                target_type : list, default=None
                    target variable type, either 'category' or 'number'
            attributes:
                data : pandas DataFrame
                    independent variables returned as a pandas DataFrame
                target : Pandas Series
                    dependent variable returned as a Pandas Series
        """
        self.remove_features = remove_features
        self.target = data[target].squeeze() if target is not None else None
        self.data = (
            data.drop(self.remove_features + [self.target.name], axis=1)
            if target is not None
            else data.drop(self.remove_features, axis=1)
        )
        self.identify_as_continuous = identify_as_continuous
        self.identify_as_boolean = identify_as_boolean
        self.identify_as_count = identify_as_count
        self.identify_as_date = identify_as_date
        self.identify_as_nominal = identify_as_nominal
        self.identify_as_ordinal = identify_as_ordinal
        self.ordinal_encodings = ordinal_encodings
        self.identify_as_string = identify_as_string
        self.target_type = target_type

        if self.identify_as_ordinal is not None and self.ordinal_encodings is None:
            warnings.warn("Recommendation - Ordinal column names passed to 'identify_as_ordinal' variable but, no ordinal encoding instructions pass to 'ordinal_encodings' variable. It is recommended to pass a dictionary containing ordinal column names as keys and lists containing the preferred order of encoding as values", UserWarning)

        # execute method feature_by_type_capture
        self.data = PreserveMetaData(self.data)
        self.capture_mlm_dtypes()

        # encode the target column if there is one
        if self.target is not None and self.target_type == "category":
            self.encode_target()

    def capture_mlm_dtypes(self):
        """
        documentation:
            description:
                determine feature type for each feature as being object, number
                or bool.
        """
        ### populate mlm_dtypes dictionary with feature type label for each feature
        self.data.mlm_dtypes = {}

        ### boolean
        # mlmachine dtype capture
        if isinstance(self.identify_as_boolean, list):
            self.data.mlm_dtypes["lean"] = self.identify_as_boolean
        elif not isinstance(self.identify_as_boolean, list) and self.identify_as_boolean is not None:
            raise AttributeError ("Variable passed to identify_as_boolean is not a list. Provide a list of column names, provide None or allow identify_as_boolean to default to None.")
        elif self.identify_as_boolean is None:
            self.data.mlm_dtypes["boolean"] = []

        # Pandas dtype
        for column in self.data.mlm_dtypes["boolean"]:
            self.data[column] = self.data[column].astype("boolean")

        ### nominal category
        # mlmachine dtype capture
        if isinstance(self.identify_as_nominal, list):
            self.data.mlm_dtypes["nominal"] = self.identify_as_nominal
        elif not isinstance(self.identify_as_nominal, list) and self.identify_as_nominal is not None:
            raise AttributeError ("Variable passed to identify_as_nominal is not a list. Provide a list of column names, provide None or allow identify_as_nominal to default to None.")
        elif self.identify_as_nominal is None:
            self.data.mlm_dtypes["nominal"] = []

        # Pandas dtype
        for column in self.data.mlm_dtypes["nominal"]:
            self.data[column] = self.data[column].astype("category")

        ### ordinal category
        # mlmachine dtype capture
        if isinstance(self.identify_as_ordinal, list):
            self.data.mlm_dtypes["ordinal"] = self.identify_as_ordinal
        elif not isinstance(self.identify_as_ordinal, list) and self.identify_as_ordinal is not None:
            raise AttributeError ("Variable passed to identify_as_ordinal is not a list. Provide a list of column names, provide None or allow identify_as_ordinal to default to None.")
        elif isinstance(self.ordinal_encodings, dict):
            self.data.mlm_dtypes["ordinal"] = list(self.ordinal_encodings.keys())
        elif self.identify_as_ordinal is None and self.ordinal_encodings is None:
            self.data.mlm_dtypes["ordinal"] = []

        # Pandas dtype
        if isinstance(self.ordinal_encodings, dict):
            for column, order in self.ordinal_encodings.items():
                category_type = CategoricalDtype(categories=order, ordered=True)
                self.data[column] = self.data[column].astype(category_type)

        for column in self.data.mlm_dtypes["ordinal"]:
            self.data[column] = self.data[column].astype("category")

        ### continuous
        # mlmachine dtype capture
        if isinstance(self.identify_as_continuous, list):
            self.data.mlm_dtypes["continuous"] = self.identify_as_continuous
        elif not isinstance(self.identify_as_continuous, list) and self.identify_as_continuous is not None:
            raise AttributeError ("Variable passed to identify_as_continuous is not a list. Either provider a list of column names, provide None or allow identify_as_continuous to default to None.")
        elif self.identify_as_continuous is None:
            self.data.mlm_dtypes["continuous"] = []

        # Pandas dtype
        for column in self.data.mlm_dtypes["continuous"]:
            self.data[column] = self.data[column].astype("float64")

        ### count
        # mlmachine dtype capture
        if isinstance(self.identify_as_count, list):
            self.data.mlm_dtypes["count"] = self.identify_as_count
        elif not isinstance(self.identify_as_count, list) and self.identify_as_count is not None:
            raise AttributeError ("Variable passed to identify_as_count is not a list. Provide a list of column names, provide None or allow identify_as_count to default to None.")
        elif self.identify_as_count is None:
            self.data.mlm_dtypes["count"] = []

        # Pandas dtype
        for column in self.data.mlm_dtypes["count"]:
            try:
                self.data[column] = self.data[column].astype("int64")
            except ValueError:
                self.data[column] = self.data[column].astype("float64")

        ### string
        # mlmachine dtype capture
        if isinstance(self.identify_as_string, list):
            self.data.mlm_dtypes["string"] = self.identify_as_string
        elif not isinstance(self.identify_as_string, list) and self.identify_as_string is not None:
            raise AttributeError ("Variable passed to identify_as_string is not a list. Provide a list of column names, provide None or allow identify_as_string to default to None.")
        elif self.identify_as_string is None:
            self.data.mlm_dtypes["string"] = []

        # Pandas dtype
        for column in self.data.mlm_dtypes["string"]:
            self.data[column] = self.data[column].astype("string")

        ### date
        # mlmachine dtype capture
        if isinstance(self.identify_as_date, list):
            self.data.mlm_dtypes["date"] = self.identify_as_date
        elif not isinstance(self.identify_as_date, list) and self.identify_as_date is not None:
            raise AttributeError ("Variable passed to identify_as_date is not a list. Provide a list of column names, provide None or allow identify_as_date to default to None.")
        elif self.identify_as_date is None:
            self.data.mlm_dtypes["date"] = []

        # Pandas dtype
        for column in self.data.mlm_dtypes["date"]:
            self.data[column] = self.data[column].astype("datetime64[ns]")

        ### untracked columns
        # compile single list of features that have already been categorized
        tracked_columns = [i for i in sum(self.data.mlm_dtypes.values(), [])]

        # iterate through untracked columns and attempt mlmachine dtype identification
        for column in [i for i in self.data.columns if i not in tracked_columns]:

            # capture column statistics and characteristics
            try:
                value_mean = np.mean(self.data[column].dropna())
                value_std = np.std(self.data[column].dropna())
            except TypeError:
                pass

            zeros_and_ones = (self.data[column].eq(0) | self.data[column].eq(1)).sum()
            unique_values = len(np.unique(self.data[column].dropna()))

            #
            if is_object_dtype(self.data[column]) \
                or is_string_dtype(self.data[column]) \
                or is_categorical_dtype(self.data[column]):

                self.data.mlm_dtypes["nominal"].append(column)
                self.data[column] = self.data[column].astype("category")

            #
            elif is_datetime64_any_dtype(self.data[column]):

                self.data.mlm_dtypes["date"].append(column)
                self.data[column] = self.data[column].astype("datetime64[ns]")

            #
            elif is_bool_dtype(self.data[column]):

                self.data.mlm_dtypes["boolean"].append(column)
                self.data[column] = self.data[column].astype("boolean")

            #
            elif is_string_dtype(self.data[column]):

                self.data.mlm_dtypes["string"].append(column)
                self.data[column] = self.data[column].astype("string")

            #
            elif is_numeric_dtype(self.data[column]):

                #
                if value_std/value_mean > 2 and value_std > 5:
                    self.data.mlm_dtypes["continuous"].append(column)
                    self.data[column] = self.data[column].astype("float64")

                # if column contains only 0's and 1's, then boolean
                elif self.data[column].astype("float").apply(float.is_integer).all() \
                    and zeros_and_ones == self.data.shape[0]:

                    self.data.mlm_dtypes["boolean"].append(column)
                    self.data[column] = self.data[column].astype("boolean")

                #
                elif self.data[column].astype("float").apply(float.is_integer).all() \
                    and zeros_and_ones != self.data.shape[0]:

                    self.data.mlm_dtypes["count"].append(column)
                    self.data[column] = self.data[column].astype("int64")

                #
                else:
                    self.data.mlm_dtypes["continuous"].append(column)
                    self.data[column] = self.data[column].astype("float64")


            # all else are category
            else:
                self.data.mlm_dtypes["nominal"].append(column)
                self.data[column] = self.data[column].astype("category")

        ### sort lists within dictionary
        self.data.mlm_dtypes = {x: sorted(self.data.mlm_dtypes[x]) for x in self.data.mlm_dtypes.keys()}

        ###
        self.data.mlm_dtypes["category"] = self.data.mlm_dtypes["boolean"] + self.data.mlm_dtypes["nominal"] + self.data.mlm_dtypes["ordinal"]

    def update_dtypes(self, columns_to_drop=None):
        """
        documentation:
            description:
                update mlm_dtypes dictionary to include new columns. ensures new object columns
                in dataset have the dtype "category". optionally drops specific columns from the dataset
                and mlm_dtypes.
            parameters:
                columns_to_drop : list, default=None
                    columns to drop from output dataset(s)/
        """
        ### updates to mlm_dtypes with new columns and drop any specified removals
        # capture current columns
        current_columns = self.data.columns.tolist()

        # capture current state of mlm_dtypes
        old_mlm_dtypes = copy.deepcopy(self.data.mlm_dtypes)

        # remove any columns listed in columns_to_drop
        if columns_to_drop is not None:
            current_columns = [x for x in current_columns if x not in columns_to_drop]

        # remove any columns_to_drop from mlm_dtypes
        for k in self.data.mlm_dtypes.keys():
            self.data.mlm_dtypes[k] = [x for x in self.data.mlm_dtypes[k] if x in current_columns]

        # remove any columns listed in columns_to_drop from the main dataset
        if columns_to_drop is not None:
            try:
                # preserve mlm_dtypes
                self.meta_mlm_dtypes = self.data.mlm_dtypes
                self.data = self.data.drop(columns_to_drop, axis=1)

                # add back mlm_dtypes
                self.data.mlm_dtypes = self.meta_mlm_dtypes
            except KeyError:
                pass

        ### capture nominal column / value pairs
        self.nominal_column_values = {}
        for column in self.data.mlm_dtypes["nominal"]:
            self.nominal_column_values[column] = list(self.data[column].dropna().unique())

        ### add any currently untracked column to mlm_dtypes and set dtype in main dataset
        # compile single list of features that have already been categorized
        tracked_columns = [i for i in sum(self.data.mlm_dtypes.values(), [])]
        untracked_columns = list(set(current_columns).difference(tracked_columns))

        for column in untracked_columns:

            # capture column statistics
            zeros_and_ones = (self.data[column].eq(0) | self.data[column].eq(1)).sum()
            unique_values = len(np.unique(self.data[column].dropna()))

            try:
                value_mean = np.mean(self.data[column].dropna())
                value_std = np.std(self.data[column].dropna())
            except TypeError:
                value_mean = 0.01
                value_std = 0.01
                pass

            # # if first portion of column name is previously identified as a category column
            # if len(column.split("_")) > 1 \
            #     and column.split("_")[0] in old_mlm_dtypes["category"] \
            #     and column.split("_")[1] in self.nominal_column_values["category"]:

            #     self.data.mlm_dtypes["nominal"].append(column)
            #     self.data[column] = self.data[column].astype("boolean")

            # if column contains non-numeric values, default to nominal mlm_dtype
            try:
                self.data[column].astype("float").apply(float.is_integer).all()
            except ValueError:
                self.data.mlm_dtypes["nominal"].append(column)
                self.data[column] = self.data[column].astype("category")
                continue

            # column dtype is object or categorical, contains only integers, and not just 0's and 1's
            if (is_object_dtype(self.data[column]) or is_categorical_dtype(self.data[column])) \
                and self.data[column].astype("float").apply(float.is_integer).all() \
                and zeros_and_ones != self.data.shape[0]:

                self.data.mlm_dtypes["ordinal"].append(column)
                self.data[column] = self.data[column].astype("category")


            # if column name suffix indicates that is is a BoxCox or YeoJohnson transformed column
            elif column.endswith(("_BoxCox","_YeoJohnson")):

                self.data.mlm_dtypes["continuous"].append(column)
                self.data[column] = self.data[column].astype("float64")

            # if column name suffix suggests a continuous feature type
            elif column.endswith(("_target_encoded", "_woe_encoded", "_catboost_encoded")):

                self.data.mlm_dtypes["continuous"].append(column)
                self.data[column] = self.data[column].astype("float64")

            # if column name suffix suggests a count feature type
            elif column.endswith(("_count_encoded")):

                self.data.mlm_dtypes["count"].append(column)
                self.data[column] = self.data[column].astype("int64")

            # if column name suffix suggests an ordinal category feature type
            elif column.endswith(("_ordinal_encoded")):

                self.data.mlm_dtypes["ordinal"].append(column)
                self.data[column] = self.data[column].astype("category")

            # if column name suffix suggests an ordinal category feature type
            elif "_binned_" in column:

                self.data.mlm_dtypes["ordinal"].append(column)
                self.data[column] = self.data[column].astype("category")

            # if column name suffix suggests a nominal category feature type
            elif column.endswith(("_binary_encoded")):

                self.data.mlm_dtypes["nominal"].append(column)
                self.data[column] = self.data[column].astype("category")

            # if column name contains "*", and the name split at "*" returns a list of len == 2, and
            # both parts of the column name are a previously identified continuous or count column
            elif "*" in column \
                and len(column.split("*")) == 2 \
                and (column.split("*")[0] in old_mlm_dtypes["continuous"] or column.split("*")[0] in old_mlm_dtypes["count"]) \
                and (column.split("*")[1] in old_mlm_dtypes["continuous"] or column.split("*")[1] in old_mlm_dtypes["count"]):

                self.data.mlm_dtypes["continuous"].append(column)
                self.data[column] = self.data[column].astype("float64")

            # if column name contains "*", and the name split at "*" returns a list of len == 2, and
            # both parts of the column name are a previously identified continuous or count column
            elif column.endswith(("^2")) \
                and (column.split("^")[0] in old_mlm_dtypes["continuous"] or column.split("^")[0] in old_mlm_dtypes["count"]):

                self.data.mlm_dtypes["continuous"].append(column)
                self.data[column] = self.data[column].astype("float64")

            #
            elif value_std/value_mean > 2 and value_std > 5:

                self.data.mlm_dtypes["continuous"].append(column)
                self.data[column] = self.data[column].astype("float64")

            # if column contains only 0's and 1's, and is not a count column
            elif len(column.split("_")) > 1 \
                and column.split("_")[0] not in old_mlm_dtypes["count"] \
                and self.data[column].astype("float").apply(float.is_integer).all() \
                and zeros_and_ones == self.data.shape[0]:

                self.data.mlm_dtypes["nominal"].append(column)
                self.data[column] = self.data[column].astype("category")

            # if first portion of column name is previously identified as a category column
            elif len(column.split("_")) > 1 \
                and column.split("_")[0] in old_mlm_dtypes["nominal"] \
                and column.split("_")[1] in self.nominal_column_values["category"]:

                self.data.mlm_dtypes["nominal"].append(column)
                self.data[column] = self.data[column].astype("category")

            #
            elif self.data[column].astype("float").apply(float.is_integer).all() \
                and zeros_and_ones != self.data.shape[0]:

                self.data.mlm_dtypes["count"].append(column)
                self.data[column] = self.data[column].astype("int64")

            # all else are nominal
            else:

                self.data.mlm_dtypes["nominal"].append(column)
                self.data[column] = self.data[column].astype("category")

        # preserve mlm_dtypes
        self.meta_mlm_dtypes = self.data.mlm_dtypes

        # sort columns alphabetically by name
        self.data = self.data.sort_index(axis=1)

        # add back mlm_dtypes
        self.data.mlm_dtypes = self.meta_mlm_dtypes

        # add helper key / value pairs
        self.data.mlm_dtypes["category"] = self.data.mlm_dtypes["boolean"] + self.data.mlm_dtypes["nominal"] + self.data.mlm_dtypes["ordinal"]
        self.data.mlm_dtypes["number"] = self.data.mlm_dtypes["continuous"] + self.data.mlm_dtypes["count"]

        self.data.mlm_dtypes = {x:sorted(self.data.mlm_dtypes[x]) for x in self.data.mlm_dtypes.keys()}

    def encode_target(self, reverse=False):
        """
        documentation:
            description:
                encode object target column and store as a Pandas Series, where
                the name is the name of the feature in the original dataset.
            parameters:
                reverse : bool, default=False
                    reverses encoding of target variables back to original variables.
        """
        # encode label
        self.le_ = preprocessing.LabelEncoder()

        # store as a named Pandas Series
        self.target = pd.Series(
            self.le_.fit_transform(self.target.values.reshape(-1)),
            name=self.target.name,
            index=self.target.index,
        )

        print(">>> category label encoding\n")
        for orig_lbl, enc_lbl in zip(
            np.sort(self.le_.classes_), np.sort(np.unique(self.target))
        ):
            print("\t{} --> {}".format(orig_lbl, enc_lbl))
        print()

        # reverse the label encoding and overwrite target variable with the original data
        if reverse:
            self.target = self.le_.inverse_transform(self.target)

    def recombine_data(self, data=None, target=None):
        """
        documentation:
            description:
                helper function for recombining the features in the 'data' variable
                and the 'target' variable into one pandas DataFrame.
            parameters:
                data : pandas DataFrame, default=None
                    pandas DataFrame containing independent variables. if left as none,
                    the feature dataset provided to machine during instantiation is used.
                target : Pandas Series, default=None
                    Pandas Series containing dependent target variable. if left as none,
                    the target dataset provided to machine during instantiation is used.
            return:
                df : pandas DataFrame
                    pandas DataFrame containing combined independent and dependent variables.
        """
        # use data/target provided during instantiation if left unspecified
        if data is None:
            data = self.data
        if target is None:
            target = self.target

        df = data.merge(target, left_index=True, right_index=True)
        return df


class PreserveMetaData(pd.DataFrame):

    _metadata = ["mlm_dtypes"]

    @property
    def _constructor(self):
        return PreserveMetaData


def train_test_df_compile(data, target_col, valid_size=0.2, random_state=1):
    """
    description:
        intakes a single dataset and returns a training dataset and a validation dataset
        stored in pandas DataFrames.
    parameters:
        data: pandas DataFrame or array
            dataset to be deconstructed into train and test sets.
        target_col : string
            name of target column in data parameter
        valid_size : float, default = 0.2
            proportion of dataset to be set aside as "unseen" test data.
        random_state : int
            random number seed
    returns:
        df_train : pandas DataFrame
            training dataset to be passed into mlmachine pipeline
        df_valid : pandas DataFrame
            validation dataset to be passed into mlmachine pipeline
    """
    if isinstance(data, pd.DataFrame):
        y = data[target_col]
        x = data.drop([target_col], axis=1)

    x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
        x, y, test_size=valid_size, random_state=1, stratify=y
    )

    df_train = x_train.merge(y_train, left_index=True, right_index=True)
    df_valid = x_valid.merge(y_valid, left_index=True, right_index=True)

    return df_train, df_valid
