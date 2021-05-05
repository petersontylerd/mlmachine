import matplotlib.pyplot as plt

import abc
import inspect
from time import gmtime, strftime

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    f_classif,
    f_regression,
    VarianceThreshold,
    SelectFromModel,
    SelectKBest,
)
from sklearn.model_selection import (
    KFold,
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    IsolationForest,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import (
    Lasso,
    Ridge,
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    SGDRegressor,
)
from sklearn.feature_selection import RFE
from sklearn.kernel_ridge import KernelRidge
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.exceptions import NotFittedError
from sklearn.base import clone
from sklearn.metrics import get_scorer, make_scorer

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import catboost

from prettierplot.plotter import PrettierPlot
from prettierplot import style

from mlxtend.feature_selection import SequentialFeatureSelector

from ..model.tune.bayesian_optim_search import BasicModelBuilder


class FeatureSelector:
    """
    Documentation:

        ---
        Description:
            Evaluate feature importance using several different feature selection techniques,
            including f_score, variance, recursive feature elimination, forward sequential selection,
            backward sequential selection, and correlation to target. Iterates through a list of
            estimators for each technique where applicable. Also includes methods for performing
            cross-validation-driven feature selection and visualization of the results.

        ---
        Parameters:
            data : Pandas DataFrame, default=None
                Pandas DataFrame containing independent variables.
            target : Pandas Series, default=None
                Pandas Series containing  target variable.
            estimators : list of strings, sklearn api objects or instantiated models
                List of estimators to cycle through when executing relevant feature importance
                techniques.
            classification : bool, default=True
                Conditional controlling whether FeatureSelector operates in classification mode
                or regression mode.
    """

    def __init__(self, data, target, estimators, classification=True):
        self.data = data
        self.target = target
        self.estimators = estimators
        self.classification = classification

    def feature_selector_suite(self, sequential_scoring=None, sequential_n_folds=0, rank=False, add_stats=False,
                                n_jobs=1, save_to_csv=False, run_variance=True, run_importance=True, run_rfe=True,
                                run_corr=True, run_f_score=True, run_sfs=True, run_sbs=True):
        """
        Documentation:

            ---
            Description:
                Run multiple feature selections processes and aggregate results. Optionally rank feature
                importance values. Optionally generate summary statistics on ranks.

            ---
            Parameters:
                sequential_scoring : str, sklearn scoring function, or list of these objects, default=None
                    Scoring metric for sequential feature selector algorithms. If list is provided,
                    algorithm is run for each scoring metric. If None, accuracy is used for classifiers
                    and r2 is used for regressors.
                sequential_n_folds : int, default=0
                    Number of folds to use in cross-validation procedure for sequential feature selector
                    algorithms. If 0 is passed, no cross-validation is performed.
                rank : bool, default=False
                    Conditional controlling whether to overwrite feature importance values with feature
                    importance rank.
                add_stats : bool, default=False
                    Add row-wise summary statistics for feature importance ranking columns. Requires feature
                    importance values to be represented as ranks. If raw feature importance values are provided,
                    values are automatically converted to ranks.
                n_jobs : int, default=1
                    Number of workers to deploy upon execution, if applicable. If estimator does not
                    have an n_jobs parameter, this is ignored.
                save_to_csv : bool, default=Fakse
                    Conditional controlling whether or not the feature selection summary results
                    are saved to a csv file.
                run_variance : bool, default=True
                    Conditional controlling whether or not the variance method is executed.
                run_importance : bool, default=True
                    Conditional controlling whether or not the feature importance method is executed.
                run_rfe : bool, default=True
                    Conditional controlling whether or not the recursive feature elimination method
                    is executed.
                run_corr : bool, default=True
                    Conditional controlling whether or not the correlation to target method is executed.
                run_f_score : bool, default=True
                    Conditional controlling whether or not the F-score method is executed.
                run_sfs : bool, default=True
                    Conditional controlling whether or not the sequential forward feature selection method
                    is executed.
                run_sbs : bool, default=True
                    Conditional controlling whether or not the sequential backward feature selection method
                    is executed.
        """
        # run each feature importance method, if method's parameter is not set to False
        self.results_variance = self.feature_selector_variance(rank=rank) if run_variance else None
        self.results_importance = self.feature_selector_importance(rank=rank, n_jobs=n_jobs) if run_importance else None
        self.results_rfe = self.feature_selector_rfe(n_jobs=n_jobs) if run_rfe else None
        self.results_forward = self.feature_selector_forward_sequential(
                                                            n_jobs=n_jobs,
                                                            scoring=sequential_scoring,
                                                            n_folds=sequential_n_folds,
                                                        ) if run_sfs else None
        self.results_backward = self.feature_selector_backward_sequential(
                                                            n_jobs=n_jobs,
                                                            scoring=sequential_scoring,
                                                            n_folds=sequential_n_folds,
                                                        ) if run_sbs else None
        self.results_corr = self.feature_selector_corr(rank=rank) if run_corr else None
        if self.classification:
            self.results_f_score = self.feature_selector_f_score_class(rank=rank) if run_f_score else None
        else:
            self.results_f_score = self.feature_selector_f_score_reg(rank=rank) if run_f_score else None

        # collect resulting DataFrames in a list
        results = [
            self.results_f_score,
            self.results_variance,
            self.results_corr,
            self.results_rfe,
            self.results_importance,
            self.results_forward,
            self.results_backward,
        ]

        # remove any None values from the DataFrame list
        results = [result for result in results if result is not None]

        # concatenate feature importance summaries
        feature_selector_summary = pd.concat(results, join="inner", axis=1)

        # optionally add summary statistics columns
        if add_stats:
            feature_selector_summary = self.feature_selector_stats(feature_selector_summary)

        # optionally save feature_selector_summary to csv
        if save_to_csv:
            feature_selector_summary.to_csv(
                "feature_selection_summary_{}.csv".format(
                    strftime("%y%m%d%H%M", gmtime())
                ),
                columns=feature_selector_summary.columns,
            )
        return feature_selector_summary

    def feature_selector_f_score_class(self, rank=False):
        """
        Documentation:

            ---
            Description:
                Calculate f-values and p-values for each feature in the context of a
                classification problem.

            ---
            Parameters:
                rank : bool, default=False
                    Conditional controlling whether to overwrite feature importance
                    values with feature importance rank.
        """
        # calculate f-values and p-values
        univariate = f_classif(self.data, self.target)

        # parse data into dictionary
        feature_dict = {}
        feature_dict["f_value"] = univariate[0]
        feature_dict["p_value"] = univariate[1]

        # load data into Pandas DataFrame
        feature_selector_summary = pd.DataFrame(data=feature_dict, index=self.data.columns)

        # optionally overwrite values with rank
        if rank:
            feature_selector_summary = self.apply_ranks(feature_selector_summary)

        return feature_selector_summary

    def feature_selector_f_score_reg(self, rank=False):
        """
        Documentation:

            ---
            Description:
                Calculate f-values and p-values for each feature in the context of a
                regression problem.

            ---
            Parameters:
                rank : bool, default=False
                    Conditional controlling whether to overwrite feature importance values
                    with feature importance rank.
        """
        # calculate f-values and p-values
        univariate = f_regression(self.data, self.target)

        # parse data into dictionary
        feature_dict = {}
        feature_dict["f_value"] = univariate[0]
        feature_dict["p_value"] = univariate[1]

        # load data into Pandas DataFrame
        feature_selector_summary = pd.DataFrame(data=feature_dict, index=self.data.columns)

        # optionally overwrite values with rank
        if rank:
            feature_selector_summary = self.apply_ranks(feature_selector_summary)

        return feature_selector_summary

    def feature_selector_variance(self, rank=False):
        """
        Documentation:

            ---
            Description:
                Calculate variance for each feature.

            --
            Parameters:
                rank : bool, default=False
                    Conditional controlling whether to overwrite feature importance values
                    with feature importance rank.
        """
        # calculate variance
        var_importance = VarianceThreshold()
        var_importance.fit(self.data)

        # load data into Pandas DataFrame
        feature_selector_summary = pd.DataFrame(
            var_importance.variances_,
            index=self.data.columns,
            columns=["variance{}".format("_rank" if rank else "")],
        )

        # optionally overwrite values with rank
        if rank:
            feature_selector_summary = self.apply_ranks(feature_selector_summary)

        return feature_selector_summary

    def feature_selector_importance(self, rank=False, add_stats=False, n_jobs=1):
        """
        Documentation:

            ---
            Description:
                for each estimator, for each feature, calculate feature importance.

            ---
            Parameters:
                rank : bool, default=False
                    Conditional controlling whether to overwrite feature importance values
                    with feature importance rank.
                add_stats : bool, default=False
                    Add row-wise summary statistics for feature importance ranking columns.
                    Requires feature importance values to be represented as ranks. If raw
                    feature importance values are provided, values are automatically converted
                    to ranks.
                n_jobs : int, default=1
                    Number of workers to deploy upon execution, if applicable. If estimator does not
                    have an n_jobs parameter, this is ignored.
        """
        feature_dict = {}

        # iterate through all provided estimators
        for estimator in self.estimators:

            # create model, estimator name to be used as column header in DataFrame
            model, estimator_name = self.model_type_check(estimator=estimator, n_jobs=n_jobs)

            # append suffix to estimator name
            estimator_name =  estimator_name + "_feature_importance"

            ## return feature importance values
            try:
                # if the model has a feature_importances_ attribute
                feature_dict[estimator_name] = model.feature_importances_(self.data.values, self.target)
            except NotFittedError:
                # otherwise, fit model and retrieve resulting feature importance values
                model.fit(self.data.values, self.target)
                feature_dict[estimator_name] = model.feature_importances_
            except AttributeError:
                # otherwise, continue
                continue

        # store data in DataFrame, using column names for the index
        feature_selector_summary = pd.DataFrame(feature_dict, index=self.data.columns)

        # optionally overwrite values with rank
        if rank:
            feature_selector_summary = self.apply_ranks(feature_selector_summary)

        # add summary statistics columns
        if add_stats:
            feature_selector_summary = self.feature_selector_stats(feature_selector_summary)

        return feature_selector_summary

    def feature_selector_rfe(self, add_stats=False, n_jobs=1):
        """
        Documentation:

            ---
            Description:
                For each estimator, perform recursive feature elimination with a step size of 1,
                and use the step in which each feature is removed to describe relative importance.

            ---
            Parameters:
                add_stats : bool, default=False
                    Add row-wise summary statistics for feature importance ranking columns.
                    Requires feature importance values to be represented as ranks. If raw
                    feature importance values are provided, values are automatically converted
                    to ranks.
                n_jobs : int, default=1
                    Number of workers to deploy upon execution, if applicable. If estimator does not
                    have an n_jobs parameter, this is ignored.
        """
        #
        feature_dict = {}

        # iterate through all provided estimators
        for estimator in self.estimators:

            # create model, estimator name to be used as column header in DataFrame
            model, estimator_name = self.model_type_check(estimator=estimator, n_jobs=n_jobs)

            # append suffix to estimator name
            estimator_name =  estimator_name + "_rfe_rank"

            # create exhaustive, single-step recursive feature elimination object
            rfe = RFE(
                    estimator=model.custom_model if hasattr(model, "custom_model") else model,
                    n_features_to_select=1,
                    step=1,
                    verbose=0
                )

            try:
                # fit RFE object and retrieve ranking
                rfe.fit(self.data, self.target)
                feature_dict[estimator_name] = rfe.ranking_
            except (RuntimeError, KeyError):
                continue

        # store data in DataFrame, using column names for the index
        feature_selector_summary = pd.DataFrame(feature_dict, index=self.data.columns)

        # add summary statistics columns
        if add_stats:
            feature_selector_summary = self.feature_selector_stats(feature_selector_summary)

        return feature_selector_summary

    def feature_selector_backward_sequential(self, scoring, n_folds=0, add_stats=False, n_jobs=1):

        """
        Documentation:

            ---
            Description:
                For each estimator, recursively remove features one at a time, capturing
                the step in which each feature is removed.

            ---
            Parameters:
                scoring : str, sklearn scoring function, or list of these objects
                    Scoring metric for sequential feature selector algorithm. If list is provided,
                    algorithm is run for each scoring metric.
                n_folds : int, default=0
                    Number of folds to use in cross-validation procedure. If 0 is passed,
                    no cross-validation is performed.
                add_stats : bool, default=False
                    Add row-wise summary statistics for feature importance ranking columns.
                    Requires feature importance values to be represented as ranks. If raw
                    feature importance values are provided, values are automatically converted
                    to ranks.
                n_jobs : int, default=1
                    Number of workers to deploy upon execution, if applicable. If estimator does not
                    have an n_jobs parameter, this is ignored.
        """
        # if scorers are specified using a list
        if isinstance(scoring, list):
            # if strings, convert to a list of sklearn scorers
            if all(isinstance(metric, str) for metric in scoring):
                scoring = [get_scorer(metric) for metric in scoring]
            # if list of callables, convert to scorers
            elif all(callable(metric) for metric in scoring):
                scoring = [make_scorer(metric) for metric in scoring]

        # if scorer is specified as a single string, convert to a list containing the associated sklearn scorer
        elif isinstance(scoring, str):
            scoring = [get_scorer(scoring)]

        # if single callable is provided, convert to a list contained the associated sklearn scorer
        elif callable(scoring):
            scoring = [make_scorer(scoring)]

        # iterate through estimators and scoring metrics
        results = []

        # iterate through all provided estimators
        for estimator in self.estimators:

            # iterate through all provided scoring techniques
            for metric in scoring:

                # create model, estimator name to be used as column header in DataFrame
                model, estimator_name = self.model_type_check(estimator=estimator, n_jobs=n_jobs)

                # append suffix to estimator name
                try:
                    estimator_name =  estimator_name + "_SBS_rank_" + metric.__name__
                except AttributeError:
                    estimator_name =  estimator_name + "_SBS_rank_" + metric._score_func.__name__

                # create exhaustive sequential feature selector object
                selector = SequentialFeatureSelector(
                            model,
                            k_features=1,
                            forward=False,
                            floating=False,
                            verbose=0,
                            scoring=metric,
                            cv=n_folds,
                            clone_estimator=False,
                    )
                selector = selector.fit(self.data, self.target)

                # collect feature set used in each sequence
                feature_sets = {}
                for index in list(selector.subsets_.keys()):
                    feature_sets[index] = set(selector.subsets_[index]["feature_names"])

                # store data in DataFrame, using column names for the index
                feature_selector_estimator_summary = pd.DataFrame(columns=["feature", estimator_name])

                # iterate through feature count and feature set
                for n_features, feature_set in feature_sets.items():
                    try:

                        # identify column that is in the previous feature set but not in the current feature set
                        selected_feature = list(feature_sets[n_features] - feature_sets[n_features - 1])
                    except:

                        # for final feature set that contain only one feature, simply capture that feature
                        selected_feature = list(feature_sets[n_features])

                    # append the dropped feature with its rank to the feature selector summary DataFrame for the current estimator
                    feature_selector_estimator_summary.loc[len(feature_selector_estimator_summary)] = [selected_feature[0], n_features]
                feature_selector_estimator_summary = feature_selector_estimator_summary.set_index("feature")

                # append estimator summary to list of all results
                results.append(feature_selector_estimator_summary)

        # concatenate all results into single feature selector summary
        feature_selector_summary = pd.concat(results, join="inner", axis=1)
        return feature_selector_summary

    def feature_selector_forward_sequential(self, scoring, n_folds=0, add_stats=False, n_jobs=1):

        """
        Documentation:

            ---
            Description:
                For each estimator, recursively remove features one at a time, capturing
                the step in which each feature is removed.

            ---
            Parameters:
                scoring : str, sklearn scoring function, or list of these objects
                    Scoring metric for sequential feature selector algorithm. If list is provided,
                    algorithm is run for each scoring metric
                n_folds : int, default=0
                    Number of folds to use in cross-validation procedure. If 0 is passed,
                    no cross-validation is performed.
                add_stats : bool, default=False
                    Add row-wise summary statistics for feature importance ranking columns.
                    Requires feature importance values to be represented as ranks. If raw
                    feature importance values are provided, values are automatically converted
                    to ranks.
                n_jobs : int, default=1
                    Number of workers to deploy upon execution, if applicable. If estimator does not
                    have an n_jobs parameter, this is ignored.
        """
        # if scorers are specified using a list
        if isinstance(scoring, list):
            # if strings, convert to a list of sklearn scorers
            if all(isinstance(metric, str) for metric in scoring):
                scoring = [get_scorer(metric) for metric in scoring]
            # if list of callables, convert to scorers
            elif all(callable(metric) for metric in scoring):
                scoring = [make_scorer(metric) for metric in scoring]

        # if scorer is specified as a single string, convert to a list containing the associated sklearn scorer
        elif isinstance(scoring, str):
            scoring = [get_scorer(scoring)]

        # if single callable is provided, convert to a list contained the associated sklearn scorer
        elif callable(scoring):
            scoring = [make_scorer(scoring)]

        # iterate through estimators and scoring metrics
        results = []

        # iterate through all provided estimators
        for estimator in self.estimators:

            # iterate through all provided scoring techniques
            for metric in scoring:

                # create model, estimator name to be used as column header in DataFrame
                model, estimator_name = self.model_type_check(estimator=estimator, n_jobs=n_jobs)

                # append suffix to estimator name
                try:
                    estimator_name =  estimator_name + "_SFS_rank_" + metric.__name__
                except AttributeError:
                    estimator_name =  estimator_name + "_SFS_rank_" + metric._score_func.__name__

                # create exhaustive sequential feature selector object
                selector = SequentialFeatureSelector(
                            model,
                            k_features=self.data.shape[1],
                            forward=True,
                            floating=False,
                            verbose=0,
                            scoring=metric,
                            cv=n_folds,
                            clone_estimator=False,
                    )
                selector = selector.fit(self.data, self.target)

                # collect feature set used in each sequence
                feature_sets = {}
                for index in selector.k_feature_idx_[::-1]:
                    feature_sets[index + 1] = set(selector.subsets_[index + 1]["feature_names"])

                # store data in DataFrame, using column names for the index
                feature_selector_estimator_summary = pd.DataFrame(columns=["feature", estimator_name])

                # iterate through feature count and feature set
                for n_features, feature_set in feature_sets.items():
                    try:

                        # identify column that is in the previous feature set but not in the current feature set
                        selected_feature = list(feature_sets[n_features] - feature_sets[n_features - 1])
                    except:

                        # for final feature set that contain only one feature, simply capture that feature
                        selected_feature = list(feature_sets[n_features])

                    # append the dropped feature with its rank to the feature selector summary DataFrame for the current estimator
                    feature_selector_estimator_summary.loc[len(feature_selector_estimator_summary)] = [selected_feature[0], n_features]
                feature_selector_estimator_summary = feature_selector_estimator_summary.set_index("feature")

                # append estimator summary to list of all results
                results.append(feature_selector_estimator_summary)

        # concatenate all results into single feature selector summary
        feature_selector_summary = pd.concat(results, join="inner", axis=1)
        return feature_selector_summary

    def feature_selector_corr(self, rank=False):
        """
        Documentation:

            ---
            Description:
                For each feature, calculate absolute correlation coefficient relative to
                target dataset.

            ---
            Parameters:
                rank : bool, default=False
                    Conditional controlling whether to overwrite feature importance values with
                    feature importance rank.
        """
        # calculate absolute correlation coefficients relative to target and store in DataFrame
        feature_selector_summary = self.data.merge(self.target, left_index=True, right_index=True)
        feature_selector_summary = pd.DataFrame(feature_selector_summary.corr().abs()[self.target.name])

        # rename, sort correlation coefficient descending, and drop target row
        feature_selector_summary = feature_selector_summary.rename(columns={self.target.name: "correlation_to_target"})
        feature_selector_summary = feature_selector_summary.sort_values("correlation_to_target", ascending=False)
        feature_selector_summary = feature_selector_summary.drop(self.target.name, axis=0)

        # optionally overwrite values with rank
        if rank:
            feature_selector_summary = self.apply_ranks(feature_selector_summary)

        return feature_selector_summary

    def apply_ranks(self, feature_selector_summary):
        """
        Documentation:

            ---
            Description:
                Apply ascending or descending ranking on feature importance values.

            ---
            Parameters:
                feature_selector_summary : Pandas DataFrame
                    Pandas DataFrame with an index corresponding to feature names and columns
                    corresponding to feature importance values.
        """

        # segment feature importance techniques based on how feature importance values translate
        # to relative importance
        ascending_rank = ["p_value","rfe"]
        descending_rank = ["feature_importance","f_value","variance","correlation_to_target"]

        ## ascending
        # capture column to rank in ascending order
        apply_ascending_rank = []
        for column in ascending_rank:
            apply_ascending_rank.extend([s for s in feature_selector_summary.columns if column in s])

        # overwrite values with ascending rank
        for column in apply_ascending_rank:
            feature_selector_summary[column] = feature_selector_summary[column].rank(ascending=True, method="max")
            feature_selector_summary = feature_selector_summary.rename({column: column + "_rank"}, axis=1)

        ## descending
        # capture column to rank in descending order
        apply_descending_rank = []
        for column in descending_rank:
            apply_descending_rank.extend([s for s in feature_selector_summary.columns if column in s])

        # overwrite values with descending rank
        for column in apply_descending_rank:
            feature_selector_summary[column] = feature_selector_summary[column].rank(ascending=False, method="max")
            feature_selector_summary = feature_selector_summary.rename({column: column + "_rank"}, axis=1)
        return feature_selector_summary

    def feature_selector_stats(self, feature_selector_summary):
        """
        Documentation:

            ---
            Description:
                Add row-wise summary statistics for feature importance ranking columns. If data
                provided includes raw feature importance values, ranking will be applied
                automatically.

            ---
            Parameters:
                feature_selector_summary : Pandas DataFrame
                    Pandas DataFrame with an index corresponding to feature names and columns
                    corresponding to feature importance values or feature importance rankings.
        """
        # apply ranking if needed
        if not feature_selector_summary.columns.str.endswith("_rank").sum() == feature_selector_summary.shape[1]:
            feature_selector_summary = self.apply_ranks(feature_selector_summary)

        ## add summary stats
        # mean
        feature_selector_summary.insert(
            loc=0, column="average", value=feature_selector_summary.mean(axis=1)
        )
        # standard deviation
        feature_selector_summary.insert(
            loc=1,
            column="stdev",
            value=feature_selector_summary.iloc[:, 1:].std(axis=1),
        )

        # best rank
        feature_selector_summary.insert(
            loc=2,
            column="best",
            value=feature_selector_summary.iloc[:, 2:].min(axis=1),
        )

        # worst rank
        feature_selector_summary.insert(
            loc=3,
            column="worst",
            value=feature_selector_summary.iloc[:, 3:].max(axis=1),
        )

        # sort on mean rank, ascending
        feature_selector_summary = feature_selector_summary.sort_values("average")
        return feature_selector_summary

    def feature_selector_cross_val(self, scoring, feature_selector_summary, estimators=None, n_folds=3,
                                    step=1, n_jobs=1, verbose=False, save_to_csv=False):
        """
        Documentation:

            ---
            Description:
                Perform cross_validation for each estimator, for progressively smaller sets of features. The list
                of features is reduced by one feature on each pass. The feature removed is the least important
                feature of the remaining set. calculates both the training and test performance.

            ---
            Parameters:
                scoring : str, sklearn scoring function, or list of these objects
                    Scoring metric for cross-validation procedure. If list is provided, procedure is run for
                    each scoring metric.
                feature_selector_summary : Pandas DataFrame or str
                    Pandas DataFrame, or str of csv file location, containing  feature_selector_suite results.
                    If None, use FeatureSelector object's internal attribute.
                estimators : list of strings, sklearn api objects or instantiated models, default=None
                    List of estimators to cycle through when executing relevant feature importance techniques.
                    If None, use estimators provided to FeatureSelector object during instantiation.
                n_folds : int, default=3
                    Number of folds to use in cross-validation procedure.
                step : int, default=1
                    Number of features to remove per iteration.
                n_jobs : int, default=1
                    Number of workers to use when training the model. This parameter will be
                    ignored if the model does not have this parameter.
                verbose : bool, default=False
                    Conditional controlling whether each estimator name is printed prior to initiating
                    cross-validation procedure for that estimator.
                save_to_csv : bool, default=False
                    Conditional controlling whethor or not the cross-validation results are saved to a csv file.
        """
        # load feature selector summary
        if isinstance(feature_selector_summary, str):
            feature_selector_summary = pd.read_csv(
                feature_selector_summary, index_col=0
            )
        elif isinstance(feature_selector_summary, pd.core.frame.DataFrame):
            feature_selector_summary = feature_selector_summary
        else:
            raise AttributeError(
                "No feature_selector_summary detected. Execute one of the feature selector methods or load from .csv"
            )

        # if scorers are specified using a list
        if isinstance(scoring, list):
            # if strings, convert to a list of sklearn scorers
            if all(isinstance(metric, str) for metric in scoring):
                scoring = [get_scorer(metric) for metric in scoring]
            # if list of callables, convert to scorers
            elif all(callable(metric) for metric in scoring):
                scoring = [make_scorer(metric) for metric in scoring]

        # if scorer is specified as a single string, convert to a list containing the associated sklearn scorer
        elif isinstance(scoring, str):
            scoring = [get_scorer(scoring)]

        # if single callable is provided, convert to a list contained the associated sklearn scorer
        elif callable(scoring):
            scoring = [make_scorer(scoring)]

        # add summary stats if needed
        if not "average" in feature_selector_summary.columns:
            feature_selector_summary = self.feature_selector_stats(feature_selector_summary)

        # load estimators if needed
        if estimators is None:
            estimators = self.estimators

        # create empty DataFrame for storing all results
        self.cv_summary = pd.DataFrame(
            columns=[
                "estimator",
                "training score",
                "validation score",
                "scoring",
                "features dropped",
            ]
        )

        # perform cross validation for all estimators for each diminishing set of features
        row_ix = 0

        # iterate through all provided estimators
        for estimator in estimators:

            # optionally print the current estimator's name to track progress
            if verbose:
                print(estimator)

            # create empty DataFrame for storing results for current estimator
            cv = pd.DataFrame(
                columns=[
                    "estimator",
                    "training score",
                    "validation score",
                    "scoring",
                    "features dropped",
                ]
            )

            # create model, estimator name to be used as column header in DataFrame
            model, estimator_name = self.model_type_check(estimator=estimator, n_jobs=n_jobs)

            # iterate through scoring metrics
            for metric in scoring:

                step_increment = 0

                # iterate through each set of features
                for i in np.arange(0, feature_selector_summary.shape[0], step):

                    # collect names of columns to form feature subset
                    if i == 0:
                        top = feature_selector_summary.sort_values("average").index
                    else:
                        top = feature_selector_summary.sort_values("average").index[:-i]

                    # special-case handling of certain metrics and metric names
                    if metric == "root_mean_squared_error":
                        metric = "neg_mean_squared_error"
                        score_transform = "rmse"
                    elif metric == "root_mean_squared_log_error":
                        metric = "neg_mean_squared_log_error"
                        score_transform = "rmsle"
                    else:
                        score_transform = metric

                    # execute cross-validation procedure with current feature subset
                    scores = cross_validate(
                        estimator=model.custom_model if hasattr(model, "custom_model") else model,
                        X=self.data[top],
                        y=self.target,
                        cv=n_folds,
                        scoring=metric,
                        return_train_score=True,
                    )

                    # special-case handling of certain metrics and metric names
                    if score_transform == "rmse":
                        training = np.mean(np.sqrt(np.abs(scores["train_score"])))
                        validation = np.mean(np.sqrt(np.abs(scores["test_score"])))
                        metric_name = "root_mean_squared_error"
                    elif score_transform == "rmsle":
                        training = np.mean(np.sqrt(np.abs(scores["train_score"])))
                        validation = np.mean(np.sqrt(np.abs(scores["test_score"])))
                        metric_name = "root_mean_squared_log_error"
                    else:
                        training = np.mean(scores["train_score"])
                        validation = np.mean(scores["test_score"])
                        metric_name = metric._score_func.__name__

                    # append results to estimator's summary DataFrame
                    cv.loc[row_ix] = [
                        estimator_name,
                        training,
                        validation,
                        metric_name,
                        step_increment,
                    ]
                    step_increment += step
                    row_ix += 1

            # append estimator cross-validation summary to main summary DataFrame
            self.cv_summary = self.cv_summary.append(cv)

        # optionally save results to csv
        if save_to_csv:
            self.cv_summary.to_csv(
                "cv_summary_{}.csv".format(strftime("%y%m%d%H%M", gmtime())),
                columns=self.cv_summary.columns,
                index_label="index",
            )

        return self.cv_summary

    def feature_selector_results_plot(self, scoring, feature_selector_summary, cv_summary=None, top_sets=0,
                                    show_features=False, show_scores=None, marker_on=True, title_scale=0.7,
                                    chart_scale=15):
        """
        Documentation:

            ---
            Description:
                For each estimator, visualize the training and validation performance
                for each feature set.

            ---
            Parameters:
                scoring : str
                    Scoring metric to visualize.
                cv_summary : Pandas DataFrame or str, default=None
                    Pandas DataFrame, or str of csv file location, containing cross-validation results.
                    If None, use FeatureSelector object's internal attribute.
                feature_selector_summary : Pandas DataFrame or str
                    Pandas DataFrame, or str of csv file location.
                top_sets : int, default=0
                    Display performance for the top N feature sets.
                show_features : bool, default=False
                    Conditional controlling whether to print feature set for best validation score.
                show_scores : int or none, default=None
                    Display certain number of top features. If None, display nothing. If int, display
                    the specified number of features as a Pandas DataFrame.
                marker_on : bool, default=True
                    Conditional controlling whether to display marker for each individual score.
                title_scale : float, default=0.7
                    Controls the scaling up (higher value) and scaling down (lower value) of the size of
                    the main chart title, the x_axis title and the y_axis title.
                chart_scale : float or int, default=15
                    Chart proportionality control. Determines relative size of figure size, axis labels,
                    Chart title, tick labels, and tick marks.
        """
        # load feature selector summary
        if isinstance(feature_selector_summary, str):
            feature_selector_summary = pd.read_csv(
                feature_selector_summary, index_col=0
            )
        elif isinstance(feature_selector_summary, pd.core.frame.DataFrame):
            feature_selector_summary = feature_selector_summary
        else:
            raise AttributeError(
                "No feature_selector_summary detected. either execute method feature_selector_suite or load from .csv"
            )

        # load cv_summary if needed
        if isinstance(cv_summary, str):
            cv_summary = pd.read_csv(cv_summary, index_col=0)
        elif isinstance(cv_summary, pd.core.frame.DataFrame):
            cv_summary = cv_summary
        elif cv_summary is None:
            cv_summary = self.cv_summary

        # add summary stats if needed
        if not "average" in feature_selector_summary.columns:
            feature_selector_summary = self.feature_selector_stats(feature_selector_summary)

        # iterate through each unique estimator in cv_summary
        for estimator in cv_summary["estimator"].unique():

            print(scoring)
            print(estimator)

            # subset cv_summary by scoring method and current estimator
            cv = cv_summary[
                (cv_summary["scoring"] == scoring)
                & (cv_summary["estimator"] == estimator)
            ]

            # capture total number of features, iterations, and step size
            total_features = feature_selector_summary.shape[0]
            iters = cv.shape[0]
            step = np.ceil(total_features / iters)

            # reset index based on step size
            cv.set_index(
                keys=np.arange(0, cv.shape[0] * step, step, dtype=int), inplace=True
            )

            # optionally display top N scores, where N = show_scores
            if show_scores is not None:
                display(cv[:show_scores])

            ## capture best iteration's feature drop count and performance score
            # special handling for select metrics
            if scoring in ["root_mean_squared_error", "root_mean_squared_log_error"]:
                sort_order = True
            else:
                sort_order = False

            # retrieve number of features dropped to form best subset
            num_dropped = cv.sort_values(["validation score"], ascending=sort_order)[
                :1
            ]["features dropped"].values[0]

            # retrieve score achieved by best subset
            score = np.round(
                cv.sort_values(["validation score"], ascending=sort_order)[
                    "validation score"
                ][:1].values[0],
                5,
            )

            # display performance for the top n feature sets
            if top_sets > 0:
                display(
                    cv.sort_values(["validation score"], ascending=sort_order)[
                        :top_sets
                    ]
                )

            # optionally print feature names of features used in best subset
            if show_features:
                if num_dropped > 0:
                    features = feature_selector_summary.shape[0] - num_dropped
                    features_used = (
                        feature_selector_summary.sort_values("average")
                        .index[:features]
                        .values
                    )
                else:
                    features_used = feature_selector_summary.sort_values(
                        "average"
                    ).index.values
                print(features_used)

            # create prettierplot object
            p = PrettierPlot(chart_scale=chart_scale)

            # add canvas to prettierplot object
            ax = p.make_canvas(
                title="{}\nBest validation {} = {}\nFeatures dropped = {}".format(
                    estimator, scoring, score, num_dropped
                ),
                x_label="features removed",
                y_label=scoring,
                y_shift=0.4 if len(scoring) > 18 else 0.57,
                title_scale=title_scale,
            )

            # plot multi-line plot on canvas
            p.multi_line(
                x=cv.index,
                y=["training score", "validation score"],
                label=["training score", "validation score"],
                df=cv,
                y_units="fff",
                marker_on=marker_on,
                bbox=(1.3, 0.9),
                ax=ax,
            )
            plt.show()

    def create_cross_val_features_df(self, scoring, feature_selector_summary, cv_summary=None):
        """
        Documentation:

            ---
            Description:
                for each estimator, visualize the training and validation performance
                for each feature set.

            ---
            Parameters:
                scoring : str
                    Scoring metric to visualize.
                feature_selector_summary : Pandas DataFrame or str
                    Pandas DataFrame, or str of csv file location, containing summary of
                    feature_selector_suite results.
                cv_summary : Pandas DataFrame or str, default=None
                    Pandas DataFrame, or str of csv file location, containing cross-validation results.
                    If None, use FeatureSelector object's internal attribute.
        """
        # load feature selector summary
        if isinstance(feature_selector_summary, str):
            feature_selector_summary = pd.read_csv(
                feature_selector_summary, index_col=0
            )
        elif isinstance(feature_selector_summary, pd.core.frame.DataFrame):
            feature_selector_summary = feature_selector_summary
        else:
            raise AttributeError(
                "No feature_selector_summary detected. either execute method feature_selector_suite or load from .csv"
            )

        # load cv_summary if needed
        if isinstance(cv_summary, str):
            cv_summary = pd.read_csv(cv_summary, index_col=0)
        elif isinstance(cv_summary, pd.core.frame.DataFrame):
            cv_summary = cv_summary
        elif cv_summary is None:
            cv_summary = self.cv_summary

        # add summary stats if needed
        if not "average" in feature_selector_summary.columns:
            feature_selector_summary = self.feature_selector_stats(feature_selector_summary)

        # create empty DataFrame with feature names as index
        self.cross_val_features_df = pd.DataFrame(index=feature_selector_summary.index)

        # iterate through estimators
        for estimator in cv_summary["estimator"].unique():

            # subset cv_summary by scoring method and current estimator
            cv = cv_summary[
                (cv_summary["scoring"] == scoring)
                & (cv_summary["estimator"] == estimator)
            ]
            cv = cv.reset_index(drop=True)

            ## capture best iteration's feature drop count
            # special handling for select metrics
            if scoring in ["root_mean_squared_error", "root_mean_squared_log_error"]:
                sort_order = True
            else:
                sort_order = False

            # retrieve number of features dropped to form best subset
            num_dropped = cv.sort_values(["validation score"], ascending=sort_order)[
                :1
            ]["features dropped"].values[0]
            if num_dropped > 0:
                features = feature_selector_summary.shape[0] - num_dropped
                features_used = (
                    feature_selector_summary.sort_values("average")
                    .index[:features]
                    .values
                )
            else:
                features_used = feature_selector_summary.sort_values(
                    "average"
                ).index.values

            # create column for estimator and populate with marker
            self.cross_val_features_df[estimator] = np.nan
            self.cross_val_features_df[estimator].loc[features_used] = "x"

        # add counter and fill nans
        self.cross_val_features_df["count"] = self.cross_val_features_df.count(axis=1)
        self.cross_val_features_df = self.cross_val_features_df.fillna("")

        # add numeric index starting at 1
        self.cross_val_features_df = self.cross_val_features_df.reset_index()
        self.cross_val_features_df.index = np.arange(
            1, len(self.cross_val_features_df) + 1
        )
        self.cross_val_features_df = self.cross_val_features_df.rename(
            columns={"index": "feature"}
        )
        return self.cross_val_features_df

    def create_cross_val_features_dict(self, scoring, feature_selector_summary, cv_summary=None):
        """
        Documentation:

            ---
            Description:
                for each estimator, visualize the training and validation performance
                for each feature set.

            ---
            Parameters:
                scoring : str
                    Scoring metric to visualize.
                feature_selector_summary : Pandas DataFrame or str
                    Pandas DataFrame, or str of csv file location, containing summary of feature_selector_suite results.
                    If None, use object's internal attribute specified during instantiation.
                cv_summary : Pandas DataFrame or str, default=None
                    Pandas DataFrame, or str of csv file location, containing cross_validation results.
                    If None, use object's internal attribute specified during instantiation.
        """

        # load feature selector summary
        if isinstance(feature_selector_summary, str):
            feature_selector_summary = pd.read_csv(
                feature_selector_summary, index_col=0
            )
        elif isinstance(feature_selector_summary, pd.core.frame.DataFrame):
            feature_selector_summary = feature_selector_summary
        else:
            raise AttributeError(
                "No feature_selector_summary detected. either execute method feature_selector_suite or load from .csv"
            )

        # load cv_summary if needed
        if isinstance(cv_summary, str):
            cv_summary = pd.read_csv(cv_summary, index_col=0)
        elif isinstance(cv_summary, pd.core.frame.DataFrame):
            cv_summary = cv_summary
        elif cv_summary is None:
            raise AttributeError(
                "no cv_summary detected. either execute method feature_selector_suite or load from .csv"
            )

        # execute create_cross_val_features_df
        cross_val_features_df = self.create_cross_val_features_df(
                                                scoring=scoring,
                                                cv_summary=cv_summary,
                                                feature_selector_summary=feature_selector_summary
                                            )

        # create empty dict with feature names as index
        self.cross_val_features_dict = {}

        cross_val_features_df = cross_val_features_df.set_index("feature")
        cross_val_features_df = cross_val_features_df.drop("count", axis=1)

        # iterate through estimators
        for estimator in cross_val_features_df.columns:

            # filter to rows with an "x"
            self.cross_val_features_dict[estimator] = cross_val_features_df[
                cross_val_features_df[estimator] == "x"
            ][estimator].index

        return self.cross_val_features_dict

    def model_type_check(self, estimator, n_jobs=1):
        """
        Documentation:

            ---
            Description:
                Detects type of estimator and create model object. Also returns
                the name of the estimator.

            ---
            Parameters:
                estimator : str, sklearn api object, or instantiated model
                    Estimator to build.
                n_jobs : int, default=1
                    Number of workers to deploy upon execution, if applicable. If estimator does not
                    have an n_jobs parameter, this is ignored.
        """
        # if estimator is passed as str, eval str to create estimator
        if isinstance(estimator, str):
            estimator = eval(estimator)

        # if estimator is an api object, pass through BasicModelBuilder
        if isinstance(estimator, type) or isinstance(estimator, abc.ABCMeta):
            model = BasicModelBuilder(estimator_class=estimator, n_jobs=n_jobs)
            estimator_name = model.estimator_name

        # otherwise clone the instantiated model that was passed
        else:
            model = clone(estimator)
            estimator_name =  self.retrieve_variable_name(estimator)

        return model, estimator_name

    def retrieve_variable_name(self, variable):

        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is variable]
            if len(names) > 0:
                return names[0]
