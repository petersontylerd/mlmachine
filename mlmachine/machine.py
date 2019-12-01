import os
import sys
import importlib
import itertools
import numpy as np
import pandas as pd

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
        KFfoldTargetEncoderTrain,
        PandasFeatureUnion,
        PlayWithPandas,
        skew_summary,
        UnprocessedColumnAdder,
    )
    from .features.missing import (
        missing_col_compare,
        missing_data_dropper_all,
    )
    from .features.outlier import (
        ExtendedIsoForest,
        OutlierIQR,
        outlier_summary,
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
        ExecBayesOptimSearch,
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

    def __init__(
        self,
        data,
        remove_features=[],
        force_to_categorical=None,
        force_to_numeric=None,
        date_features=None,
        target=None,
        target_type=None,
    ):
        """
        documentation:
            description:
                __init__ handles initial processing of main data set, identification of
                select features to be removed (if any), identification of select features
                to be considered as categorical despite the feature data type(s) (if any),
                identification of select features to be considered as numerical despite
                the feature data type(s) (if any), identification of select features to be
                considered as calendar date features (if any), identification of the feature
                representing the target (if there is one) and the type of target. returns
                data frame of independent variables, series containing dependent variable
                and a dictionary that categorizes features by data type.
            parameters:
                data : pandas DataFrame
                    input data provided as a pandas DataFrame.
                remove_features : list, default = []
                    features to be completely removed from dataset.
                force_to_categorical : list, default =None
                    preidentified categorical features that would otherwise be labeled as numeric.
                force_to_numeric : list, default =None
                    preidentified numeric features that would otherwise be labeled as categorical.
                date_features : list, default =None
                    features comprised of calendar date values.
                target : list, default =None
                    name of column containing dependent variable.
                target_type : list, default =None
                    target variable type, either 'categorical' or 'numeric'
            attributes:
                data : pandas DataFrame
                    independent variables returned as a pandas DataFrame
                target : pandas series
                    dependent variable returned as a pandas series
                features_by_dtype_ : dict
                    dictionary contains keys 'numeric', 'categorical' and/or 'date'. the corresponding values
                    are lists of column names that are of that feature type.
        """
        self.remove_features = remove_features
        self.target = data[target].squeeze() if target is not None else None
        self.data = (
            data.drop(self.remove_features + [self.target.name], axis=1)
            if target is not None
            else data.drop(self.remove_features, axis=1)
        )
        self.force_to_categorical = force_to_categorical
        self.force_to_numeric = force_to_numeric
        self.date_features = date_features
        self.target_type = target_type

        # execute method feature_type_capture
        self.feature_type_capture()

        # encode the target column if there is one
        if self.target is not None and self.target_type == "categorical":
            self.encode_target()

    def feature_type_capture(self):
        """
        documentation:
            description:
                determine feature type for each feature as being categorical, numeric
                or boolean.
        """
        ### populate feature_type dictionary with feature type label for each each
        self.feature_type = {}

        # categorical features
        if self.force_to_categorical is None:
            self.feature_type["categorical"] = []
        else:
            self.feature_type["categorical"] = self.force_to_categorical

            # convert column dtype to "category"
            self.data[self.force_to_categorical] = self.data[
                self.force_to_categorical
            ].astype("category")

        # numeric features
        if self.force_to_numeric is None:
            self.feature_type["numeric"] = []
        else:
            self.feature_type["numeric"] = self.force_to_numeric

        # boolean featuers
        self.feature_type["boolean"] = []

        # compile single list of features that have already been categorized
        handled = [i for i in sum(self.feature_type.values(), [])]

        ### determine feature type for remaining columns
        for column in [i for i in self.data.columns if i not in handled]:

            # if column is numeric and contains only two unique values, set as boolean
            if (
                pd.api.types.is_numeric_dtype(self.data[column])
                and len(self.data[column].unique()) == 2
                and np.min(self.data[column].unique()) == 0
                and np.max(self.data[column].unique()) == 1
            ):
                self.feature_type["boolean"].append(column)

                # set column in dataset as categorical data type
                self.data[column] = self.data[column].astype("bool")

            # object columns set as categorical features
            elif pd.api.types.is_object_dtype(self.data[column]):
                self.feature_type["categorical"].append(column)

                # set column in dataset as categorical data type
                self.data[column] = self.data[column].astype("category")

            # numeric features
            elif pd.api.types.is_numeric_dtype(self.data[column]):
                self.feature_type["numeric"].append(column)

        # original column definitions
        self.numeric_features = self.feature_type["numeric"]
        self.categorical_features = self.feature_type["categorical"]

    def feature_type_update(self, columns_to_drop=None):
        """
        documentation:
            description:
                update feature_type dictionary to include new columns. ensures new categorical columns
                in dataset have the dtype "category". optionally drops specific columns from the dataset
                and feature_type.
            parameters:
                columns_to_drop : list, default =None
                    columns to drop from output dataset(s)/
        """
        # set column data type as "category" where approrpirate
        for column in self.data.columns:
            # if column is numeric and contains only two unique values, set as boolean
            if (
                pd.api.types.is_numeric_dtype(self.data[column])
                and not pd.api.types.is_bool_dtype(self.data[column])
                and len(self.data[column].unique()) == 2
                and not "*" in column
                and column not in self.categorical_features
            ):
                self.data[column] = self.data[column].astype("bool")
            # if column dtype is object, set as category
            elif pd.api.types.is_object_dtype(self.data[column]):
                self.data[column] = self.data[column].astype("category")

        # determine columns already being tracked
        tracked_columns = []
        for k in self.feature_type.keys():
            tracked_columns.append(self.feature_type[k])
        tracked_columns = list(set(itertools.chain(*tracked_columns)))

        # remove any columns listed in columns_to_drop
        if columns_to_drop is not None:
            # remove from tracked_columns
            tracked_columns = [x for x in tracked_columns if x not in columns_to_drop]

            # remove from feature_type
            for k in self.feature_type.keys():
                self.feature_type[k] = [
                    x for x in self.feature_type[k] if x not in columns_to_drop
                ]

            # drop columns from main dataset
            self.data = self.data.drop(columns_to_drop, axis=1)

        # update feature_type
        for column in [i for i in self.data.columns if i not in tracked_columns]:
            # categorical features
            if pd.api.types.is_bool_dtype(self.data[column]):
                self.feature_type["boolean"].append(column)
            # categorical features
            elif pd.api.types.is_categorical(self.data[column]):
                self.feature_type["categorical"].append(column)
            # numeric features
            elif pd.api.types.is_numeric_dtype(self.data[column]):
                self.feature_type["numeric"].append(column)

        # remove columns no longer in dataset from feature_type object
        for feature_type_key in self.feature_type.keys():
            for column in self.feature_type[feature_type_key]:
                if column not in self.data.columns:
                    self.feature_type[feature_type_key] = [
                        x
                        for x in self.feature_type[feature_type_key]
                        if x not in [column]
                    ]

        # sort columns alphabeticalls by name
        self.data = self.data.sort_index(axis=1)

    def encode_target(self, reverse=False):
        """
        documentation:
            description:
                encode categorical target column and store as a pandas series, where
                the name is the name of the feature in the original dataset.
            parameters:
                reverse : boolean, default=False
                    reverses encoding of target variables back to original variables.
        """
        # encode label
        self.le_ = preprocessing.LabelEncoder()

        # store as a named pandas series
        self.target = pd.series(
            self.le_.fit_transform(self.target.values.reshape(-1)),
            name=self.target.name,
            index=self.target.index,
        )

        print("******************\n_categorical label encoding\n")
        for orig_lbl, enc_lbl in zip(
            np.sort(self.le_.classes_), np.sort(np.unique(self.target))
        ):
            print("{} __> {}".format(orig_lbl, enc_lbl))

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
                data : pandas DataFrame, default =None
                    pandas DataFrame containing independent variables. if left as none,
                    the feature dataset provided to machine during instantiation is used.
                target : pandas series, default =None
                    pandas series containing dependent target variable. if left as none,
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


def train_test_compile(data, target_col, valid_size=0.2, random_state=1):
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
