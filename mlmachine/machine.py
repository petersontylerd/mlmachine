import copy
import os
import sys
import importlib
import itertools
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype

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

    def __init__(self, data, remove_features=[], identify_as_category=None, identify_as_continuous=None,
                identify_as_count=None, identify_as_date=None, target=None, target_type=None):
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
                identify_as_category : list, default=None
                    preidentified category features. columns given float64 dtype.
                identify_as_count : list, default=None
                    preidentified count features. columns given float64 dtype.
                identify_as_continuous : list, default=None
                    preidentified continuous features. columns given float64 dtype.
                identify_as_date : list, default=None
                    preidentified date features. columns given datetime64 dtype.
                target : list, default=None
                    name of column containing dependent variable.
                target_type : list, default=None
                    target variable type, either 'category' or 'number'
            attributes:
                data : pandas DataFrame
                    independent variables returned as a pandas DataFrame
                target : Pandas Series
                    dependent variable returned as a Pandas Series
                feature_by_mlm_dtype : dict
                    dictionary contains keys 'bool','continuous','count','date','nominal' and 'ordinal'. the corresponding values
                    are lists of column names that are of that feature type.
        """
        self.remove_features = remove_features
        self.target = data[target].squeeze() if target is not None else None
        self.data = (
            data.drop(self.remove_features + [self.target.name], axis=1)
            if target is not None
            else data.drop(self.remove_features, axis=1)
        )
        self.identify_as_category = identify_as_category
        self.identify_as_continuous = identify_as_continuous
        self.identify_as_count = identify_as_count
        self.identify_as_date = identify_as_date
        self.target_type = target_type

        # execute method feature_by_type_capture
        self.data = PreserveMetaData(self.data)
        self.capture_feature_by_mlm_dtype()

        # encode the target column if there is one
        if self.target is not None and self.target_type == "category":
            self.encode_target()

    def capture_feature_by_mlm_dtype(self):
        """
        documentation:
            description:
                determine feature type for each feature as being object, number
                or bool.
        """
        ### populate feature_by_mlm_dtype dictionary with feature type label for each feature
        self.data.feature_by_mlm_dtype = {}

        # category feature identification
        if self.identify_as_category is None:
            self.data.feature_by_mlm_dtype["category"] = []
        else:
            self.data.feature_by_mlm_dtype["category"] = self.identify_as_category

        # continuous feature identification
        if self.identify_as_continuous is None:
            self.data.feature_by_mlm_dtype["continuous"] = []
        else:
            self.data.feature_by_mlm_dtype["continuous"] = self.identify_as_continuous

        # discrete feature identification
        if self.identify_as_count is None:
            self.data.feature_by_mlm_dtype["count"] = []
        else:
            self.data.feature_by_mlm_dtype["count"] = self.identify_as_count

        # discrete feature identification
        if self.identify_as_date is None:
            self.data.feature_by_mlm_dtype["date"] = []
        else:
            self.data.feature_by_mlm_dtype["date"] = self.identify_as_date

        # compile single list of features that have already been categorized
        tracked_columns = [i for i in sum(self.data.feature_by_mlm_dtype.values(), [])]

        ### determine feature type for remaining columns
        for column in [i for i in self.data.columns if i not in tracked_columns]:

            # capture column statistics
            zeros_and_ones = (self.data[column].eq(0) | self.data[column].eq(1)).sum()
            unique_values = len(np.unique(self.data[column].dropna()))

            try:
                value_mean = np.mean(self.data[column].dropna())
                value_std = np.std(self.data[column].dropna())
            except TypeError:
                pass

            # column dtype is object
            if is_object_dtype(self.data[column]):
                self.data.feature_by_mlm_dtype["category"].append(column)

            #
            elif is_numeric_dtype(self.data[column]):

                #
                if unique_values > 50:
                    self.data.feature_by_mlm_dtype["continuous"].append(column)

                #
                elif value_std/value_mean > 2 and value_std > 5:
                    self.data.feature_by_mlm_dtype["continuous"].append(column)

                # if column contains only 0's and 1's
                elif self.data[column].astype("float").apply(float.is_integer).all() \
                    and zeros_and_ones == self.data.shape[0]:

                    self.data.feature_by_mlm_dtype["category"].append(column)

                #
                elif self.data[column].astype("float").apply(float.is_integer).all() \
                    and zeros_and_ones != self.data.shape[0]:

                    self.data.feature_by_mlm_dtype["count"].append(column)

                #
                else:
                    self.data.feature_by_mlm_dtype["continuous"].append(column)

            # all else are category
            else:
                self.data.feature_by_mlm_dtype["category"].append(column)

        ### set data types
        for dtype, columns in self.data.feature_by_mlm_dtype.items():
            if dtype == "category":
                for column in columns:
                    # if is_numeric_dtype(self.data[column]):
                    #     self.data[column] = self.data[column].astype("float64")
                    # else:
                    self.data[column] = self.data[column].astype("object")
            elif dtype == "continuous":
                for column in columns:
                    self.data[column] = self.data[column].astype("float64")
            elif dtype == "count":
                for column in columns:
                    self.data[column] = self.data[column].astype("float64")
            elif dtype == "date":
                for column in columns:
                    self.data[column] = self.data[column].astype("datetime64[ns]")

        ### sort lists within dictionary
        self.data.feature_by_mlm_dtype = {x:sorted(self.data.feature_by_mlm_dtype[x]) for x in self.data.feature_by_mlm_dtype.keys()}

    def update_feature_by_mlm_dtype(self, columns_to_drop=None):
        """
        documentation:
            description:
                update feature_by_mlm_dtype dictionary to include new columns. ensures new object columns
                in dataset have the dtype "category". optionally drops specific columns from the dataset
                and feature_by_mlm_dtype.
            parameters:
                columns_to_drop : list, default=None
                    columns to drop from output dataset(s)/
        """
        ### updates to feature_by_mlm_dtype with new columns and drop any specified removals
        # capture current columns
        current_columns = self.data.columns.tolist()

        # capture current state of feature_by_mlm_dtype
        old_feature_by_mlm_dtype = copy.deepcopy(self.data.feature_by_mlm_dtype)

        # remove any columns listed in columns_to_drop
        if columns_to_drop is not None:
            current_columns = [x for x in current_columns if x not in columns_to_drop]

        # update feature_by_mlm_dtype
        for k in self.data.feature_by_mlm_dtype.keys():
            self.data.feature_by_mlm_dtype[k] = [x for x in self.data.feature_by_mlm_dtype[k] if x in current_columns]

        # remove any columns listed in columns_to_drop from the main dataset
        if columns_to_drop is not None:
            try:
                # preserve feature_by_mlm_dtype
                self.meta = self.data.feature_by_mlm_dtype
                self.data = self.data.drop(columns_to_drop, axis=1)

                # add back feature_by_mlm_dtype
                self.data.feature_by_mlm_dtype = self.meta
            except KeyError:
                pass

        ### add any currently untracked column to feature_by_mlm_dtype and set dtype in main dataset
        # compile single list of features that have already been categorized
        tracked_columns = [i for i in sum(self.data.feature_by_mlm_dtype.values(), [])]
        untracked_columns = list(set(current_columns).difference(tracked_columns))

        for column in untracked_columns:

            # capture column statistics
            zeros_and_ones = (self.data[column].eq(0) | self.data[column].eq(1)).sum()
            unique_values = len(np.unique(self.data[column].dropna()))

            try:
                value_mean = np.mean(self.data[column].dropna())
                value_std = np.std(self.data[column].dropna())
            except TypeError:
                pass

            # if column contains non-numeric values
            try:
                self.data[column].astype("float").apply(float.is_integer).all()
            except ValueError:
                self.data.feature_by_mlm_dtype["category"].append(column)
                continue

            # if column name suffix indicates that it is an encoded column
            if column.endswith(("_bins","_count","_target_encoder","_woe","_binarized")):
                self.data.feature_by_mlm_dtype["category"].append(column)

            # if column name suffix indicates that is is a BoxCox or YeoJohnson transformed column
            elif column.endswith(("_BoxCox","_YeoJohnson")):
                self.data.feature_by_mlm_dtype["continuous"].append(column)

            # if column name contains "*", and the name split at "*" returns a list of len == 2, and
            # both parts of the column name are a previously identified continuous or count column
            elif "*" in column \
                and len(column.split("*")) == 2 \
                and (column.split("*")[0] in old_feature_by_mlm_dtype["continuous"] or column.split("*")[0] in old_feature_by_mlm_dtype["count"]) \
                and (column.split("*")[1] in old_feature_by_mlm_dtype["continuous"] or column.split("*")[1] in old_feature_by_mlm_dtype["count"]):

                self.data.feature_by_mlm_dtype["continuous"].append(column)

            # if column name contains "*", and the name split at "*" returns a list of len == 2, and
            # both parts of the column name are a previously identified continuous or count column
            elif "^2" in column \
                and (column.split("^")[0] in old_feature_by_mlm_dtype["continuous"] or column.split("^")[0] in old_feature_by_mlm_dtype["count"]):

                self.data.feature_by_mlm_dtype["continuous"].append(column)

            #
            elif unique_values > 50:
                self.data.feature_by_mlm_dtype["continuous"].append(column)

            #
            elif value_std/value_mean > 2 and value_std > 5:
                self.data.feature_by_mlm_dtype["continuous"].append(column)

            # column dtype is object
            elif is_object_dtype(self.data[column]):
                self.data.feature_by_mlm_dtype["category"].append(column)

            # if column contains only 0's and 1's, and is not a count column
            elif len(column.split("_")) > 1 \
                and column.split("_")[0] not in old_feature_by_mlm_dtype["count"] \
                and self.data[column].astype("float").apply(float.is_integer).all() \
                and zeros_and_ones == self.data.shape[0]:

                self.data.feature_by_mlm_dtype["category"].append(column)

            # if first portion of column name is previously identified as a category column
            elif len(column.split("_")) > 1 \
                and column.split("_")[0] in old_feature_by_mlm_dtype["category"]:

                self.data.feature_by_mlm_dtype["category"].append(column)

            #
            elif self.data[column].astype("float").apply(float.is_integer).all() \
                and zeros_and_ones != self.data.shape[0]:

                self.data.feature_by_mlm_dtype["count"].append(column)

            #
            elif is_numeric_dtype(self.data[column]):
                self.data.feature_by_mlm_dtype["continuous"].append(column)

            # all else are category
            else:
                self.data.feature_by_mlm_dtype["category"].append(column)

        ### set data types
        for dtype, columns in self.data.feature_by_mlm_dtype.items():
            if dtype == "category":
                for column in columns:
                    # if is_numeric_dtype(self.data[column]):
                    #     self.data[column] = self.data[column].astype("float64")
                    # else:
                    self.data[column] = self.data[column].astype("object")
            elif dtype == "continuous":
                for column in columns:
                    self.data[column] = self.data[column].astype("float64")
            elif dtype == "count":
                for column in columns:
                    self.data[column] = self.data[column].astype("float64")
            elif dtype == "date":
                for column in columns:
                    self.data[column] = self.data[column].astype("datetime64[ns]")

        # preserve feature_by_mlm_dtype
        self.meta = self.data.feature_by_mlm_dtype

        # sort columns alphabetically by name
        self.data = self.data.sort_index(axis=1)

        # add back feature_by_mlm_dtype
        self.data.feature_by_mlm_dtype = self.meta

        self.data.feature_by_mlm_dtype = {x:sorted(self.data.feature_by_mlm_dtype[x]) for x in self.data.feature_by_mlm_dtype.keys()}

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

    _metadata = ["feature_by_mlm_dtype"]

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
