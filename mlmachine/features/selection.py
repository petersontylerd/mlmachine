import matplotlib.pyplot as plt

from time import gmtime, strftime

import numpy as np
import pandas as pd

import sklearn.base as base
import sklearn.feature_selection as feature_selection
import sklearn.model_selection as model_selection

import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.kernel_ridge as kernel_ridge
import sklearn.naive_bayes as naive_bayes
import sklearn.neighbors as neighbors
import sklearn.svm as svm
import sklearn.tree as tree

import xgboost
import lightgbm
import catboost

from prettierplot.plotter import PrettierPlot
from prettierplot import style


from ..model.tune.bayesian_optim_search import BasicModelBuilder


class FeatureSelector:
    """
    documentation:
        description:
            evaluate feature importance using several different feature selection techniques,
            including f_score, variance, recursive feature selection, and correlation to target
            on a list of estimators. also includes methods for performing corss_validation and
            visualization of the results.p
        parameters:
            data : pandas DataFrame, default =None
                pandas DataFrame containing independent variables. if left as none,
                the feature dataset provided to machine during instantiation is used.
            target : Pandas Series, default =None
                Pandas Series containing dependent target variable. if left as none,
                the target dataset provided to machine during instantiation is used.
            estimators : list of strings or sklearn api objects.
                    list of estimators to be used.
            rank : boolean, default=True
                conditional controlling whether to overwrite values with rank of values.
            classification : boolean, default=True
                conditional controlling whether object is informed that the supervised learning
                task is a classification task.
    """

    def __init__(self, data, target, estimators, rank=True, classification=True):
        self.data = data
        self.target = target
        self.estimators = estimators
        self.rank = rank
        self.classification = classification

    def feature_selector_suite(self, save_to_csv=True):
        """
        documentation:
            description:
                run all feature selections processes and aggregate results. calculate summary
                statistics on results.
            parameters:
                save_to_csv : boolean, default=True
                    conditional controlling whethor or not the feature selection summary results
                    are saved to a csv file.
        """
        # run individual top feature processes
        self.results_variance = self.feature_selector_variance()
        self.results_importance = self.feature_selector_importance()
        self.results_rfe = self.feature_selector_rfe()
        self.results_corr = self.feature_selector_corr()
        if self.classification:
            self.results_f_score = self.feature_selector_f_score_class()
        else:
            self.results_f_score = self.feature_selector_f_score_reg()

        # combine results into single summary table
        results = [
            self.results_f_score,
            self.results_variance,
            self.results_corr,
            self.results_rfe,
            self.results_importance,
        ]
        self.feature_selector_summary = pd.concat(results, join="inner", axis=1)

        # add summary stats
        self.feature_selector_summary.insert(
            loc=0, column="average", value=self.feature_selector_summary.mean(axis=1)
        )
        self.feature_selector_summary.insert(
            loc=1,
            column="stdev",
            value=self.feature_selector_summary.iloc[:, 1:].std(axis=1),
        )
        self.feature_selector_summary.insert(
            loc=2,
            column="low",
            value=self.feature_selector_summary.iloc[:, 2:].min(axis=1),
        )
        self.feature_selector_summary.insert(
            loc=3,
            column="high",
            value=self.feature_selector_summary.iloc[:, 3:].max(axis=1),
        )

        self.feature_selector_summary = self.feature_selector_summary.sort_values(
            "average"
        )

        if save_to_csv:
            self.feature_selector_summary.to_csv(
                "feature_selection_summary_{}.csv".format(
                    strftime("%y%m%d%H%M", gmtime())
                ),
                columns=self.feature_selector_summary.columns,
                # index_label="index"
            )
        return self.feature_selector_summary

    def feature_selector_f_score_class(self):
        """
        documentation:
            description:
                for each feature, calculate f_values and p_values in the context of a
                classification probelm.
        """
        # calculate f_values and p_values
        univariate = feature_selection.f_classif(self.data, self.target)

        # parse data into dictionary
        feature_dict = {}
        feature_dict["f_value"] = univariate[0]
        feature_dict["p_value"] = univariate[1]

        # load dictionary into pandas DataFrame and rank values
        feature_df = pd.DataFrame(data=feature_dict, index=self.data.columns)

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            feature_df["f_value"] = feature_df["f_value"].rank(
                ascending=False, method="max"
            )
            feature_df["p_value"] = feature_df["p_value"].rank(
                ascending=True, method="max"
            )

        return feature_df

    def feature_selector_f_score_reg(self):
        """
        documentation:
            description:
                for each feature, calculate f_values and p_values in the context of a
                regression probelm.
        """
        # calculate f_values and p_values
        univariate = feature_selection.f_regression(self.data, self.target)

        # parse data into dictionary
        feature_dict = {}
        feature_dict["f_value"] = univariate[0]
        feature_dict["p_value"] = univariate[1]

        # load dictionary into pandas DataFrame and rank values
        feature_df = pd.DataFrame(data=feature_dict, index=self.data.columns)

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            feature_df["f_value"] = feature_df["f_value"].rank(
                ascending=False, method="max"
            )
            feature_df["p_value"] = feature_df["p_value"].rank(
                ascending=True, method="max"
            )

        return feature_df

    def feature_selector_variance(self):
        """
        documentation:
            description:
                for each feature, calculate variance.
        """
        # calculate variance
        var_importance = feature_selection.VarianceThreshold()
        var_importance.fit(self.data)

        # load data into pandas DataFrame and rank values
        feature_df = pd.DataFrame(
            var_importance.variances_, index=self.data.columns, columns=["variance"]
        )

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            feature_df["variance"] = feature_df["variance"].rank(
                ascending=False, method="max"
            )

        return feature_df

    def feature_selector_importance(self):
        """
        documentation:
            description:
                for each estimator, for each feature, calculate feature importance.
        """
        #
        feature_dict = {}
        for estimator in self.estimators:
            model = BasicModelBuilder(estimator=estimator)

            # estimator name
            if model.estimator.__module__.split(".")[1] == "sklearn":
                estimator_module = model.estimator.__module__.split(".")[0]
            else:
                estimator_module = model.estimator.__module__.split(".")[1]

            estimator_class = model.estimator.__name__
            estimator_name = estimator_module + "." + estimator_class

            # build dict
            feature_dict[
                "feature_importance "
                + estimator_name
                # "feature_importance " + model.estimator.__name__
            ] = model.feature_importances(self.data.values, self.target)

        feature_df = pd.DataFrame(feature_dict, index=self.data.columns)

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            feature_df = feature_df.rank(ascending=False, method="max")

        return feature_df

    def feature_selector_rfe(self):
        """
        documentation:
            description:
                for each estimator, recursively remove features one at a time, capturing
                the step in which each feature is removed.
        """
        #
        feature_dict = {}
        for estimator in self.estimators:
            model = BasicModelBuilder(estimator=estimator)

            # estimator name
            if model.estimator.__module__.split(".")[1] == "sklearn":
                estimator_module = model.estimator.__module__.split(".")[0]
            else:
                estimator_module = model.estimator.__module__.split(".")[1]

            estimator_class = model.estimator.__name__
            estimator_name = estimator_module + "." + estimator_class

            # recursive feature selection
            rfe = feature_selection.RFE(
                estimator=model.model, n_features_to_select=1, step=1, verbose=0
            )
            rfe.fit(self.data, self.target)
            feature_dict["RFE " + estimator_name] = rfe.ranking_
            # feature_dict["rfe " + model.estimator.__name__] = rfe.ranking_

        feature_df = pd.DataFrame(feature_dict, index=self.data.columns)

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            feature_df = feature_df.rank(ascending=True, method="max")

        return feature_df

    def feature_selector_corr(self):
        """
        documentation:
            description:
                for each feature, calculate absolute correlation coefficient relative to
                target dataset.
        """
        # calculate absolute correlation coefficients relative to target
        feature_df = self.data.merge(self.target, left_index=True, right_index=True)

        feature_df = pd.DataFrame(feature_df.corr().abs()[self.target.name])
        feature_df = feature_df.rename(columns={self.target.name: "target_correlation"})

        feature_df = feature_df.sort_values("target_correlation", ascending=False)
        feature_df = feature_df.drop(self.target.name, axis=0)

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            feature_df = feature_df.rank(ascending=False, method="max")

        return feature_df

    def feature_selector_cross_val(
        self,
        scoring,
        feature_selector_summary=None,
        estimators=None,
        n_folds=3,
        step=1,
        n_jobs=4,
        verbose=True,
        save_to_csv=True,
    ):
        """
        documentation:
            description:
                perform cross_validation for each estimator, for progressively smaller sets of features. the list
                of features is reduced by one feature on each pass. the feature removed is the least important
                feature of the remaining set. calculates both the training and test performance.
            parameters:
                scoring : list of strings
                    list containing strings for one or more performance scoring metrics.
                feature_selector_summary : pandas DataFrame or str, default =None
                    pandas DataFrame, or str of csv file location, containing summary of feature_selector_suite results.
                    if none, use object's internal attribute specified during instantiation.
                estimators : list of strings or sklearn api objects, default =None
                    list of estimators to be used. if none, use object's internal attribute specified during instantiation.
                n_folds : int, default = 3
                    number of folds to use in cross validation.
                step : int, default = 1
                    number of features to remove per iteration.
                n_jobs : int, default = 4
                    number of works to use when training the model. this parameter will be
                    ignored if the model does not have this parameter.
                verbose : boolean, default=True
                    conditional controlling whether each estimator name is printed prior to cross_validation.
                save_to_csv : boolean, default=True
                    conditional controlling whethor or not the feature selection summary results
                    are saved to a csv file.
        """
        # load results summary if needed
        if isinstance(feature_selector_summary, str):
            feature_selector_summary = pd.read_csv(
                feature_selector_summary, index_col=0
            )
        elif isinstance(feature_selector_summary, pd.core.frame.DataFrame):
            feature_selector_summary = feature_selector_summary
        elif feature_selector_summary is None:
            try:
                feature_selector_summary = self.feature_selector_summary
            except AttributeError:
                raise AttributeError(
                    "no feature_selector_summary detected. either execute method feature_selector_suite or load from csv"
                )

        # load estimators if needed
        if estimators is None:
            estimators = self.estimators

        # create empty dictionary for capturing one DataFrame for each estimator
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

        for estimator in estimators:

            if verbose:
                print(estimator)

            # instantiate default model and create empty DataFrame for capturing scores
            model = BasicModelBuilder(estimator=estimator, n_jobs=n_jobs)
            cv = pd.DataFrame(
                columns=[
                    "estimator",
                    "training score",
                    "validation score",
                    "scoring",
                    "features dropped",
                ]
            )

            # estimator name
            if model.estimator.__module__.split(".")[1] == "sklearn":
                estimator_module = model.estimator.__module__.split(".")[0]
            else:
                estimator_module = model.estimator.__module__.split(".")[1]

            estimator_class = model.estimator.__name__
            estimator_name = estimator_module + "." + estimator_class

            # iterate through scoring metrics
            for metric in scoring:

                step_increment = 0
                # iterate through each set of features
                for i in np.arange(0, feature_selector_summary.shape[0], step):
                    # collect names of top columns
                    if i == 0:
                        top = feature_selector_summary.sort_values("average").index
                    else:
                        top = feature_selector_summary.sort_values("average").index[:-i]

                    # custom metric handling
                    if metric == "root_mean_squared_error":
                        metric = "neg_mean_squared_error"
                        score_transform = "rmse"
                    elif metric == "root_mean_squared_log_error":
                        metric = "neg_mean_squared_log_error"
                        score_transform = "rmsle"
                    else:
                        score_transform = metric

                    scores = model_selection.cross_validate(
                        estimator=model.model,
                        X=self.data[top],
                        y=self.target,
                        cv=n_folds,
                        scoring=metric,
                        return_train_score=True,
                    )

                    # custom metric handling
                    if score_transform == "rmse":
                        training = np.mean(np.sqrt(np.abs(scores["train_score"])))
                        validation = np.mean(np.sqrt(np.abs(scores["test_score"])))
                        metric = "root_mean_squared_error"
                    elif score_transform == "rmsle":
                        training = np.mean(np.sqrt(np.abs(scores["train_score"])))
                        validation = np.mean(np.sqrt(np.abs(scores["test_score"])))
                        metric = "root_mean_squared_log_error"
                    else:
                        training = np.mean(scores["train_score"])
                        validation = np.mean(scores["test_score"])

                    # append results
                    cv.loc[row_ix] = [
                        estimator_name,
                        training,
                        validation,
                        metric,
                        step_increment,
                    ]
                    # cv.loc[row_ix] = [model.estimator.__name__, training, validation, metric, step_increment]
                    step_increment += step
                    row_ix += 1

            # capturing results DataFrame associated with estimator
            self.cv_summary = self.cv_summary.append(cv)

        if save_to_csv:
            self.cv_summary.to_csv(
                "cv_summary_{}.csv".format(strftime("%y%m%d%H%M", gmtime())),
                columns=self.cv_summary.columns,
                index_label="index",
            )

        return self.cv_summary

    def feature_selector_results_plot(
        self,
        metric,
        cv_summary=None,
        feature_selector_summary=None,
        top_sets=0,
        show_features=False,
        show_scores=None,
        marker_on=True,
        title_scale=0.7,
    ):
        """
        documentation:
            description:
                for each estimator, visualize the training and validation performance
                for each feature set.
            parameters:
                metric : string
                    metric to visualize.
                cv_summary : pandas DataFrame or str, default =None
                    pandas DataFrame, or str of csv file location, containing cross_validation results.
                    if none, use object's internal attribute specified during instantiation.
                feature_selector_summary : pandas DataFrame or str, default =None
                    pandas DataFrame, or str of csv file location, containing summary of feature_selector_suite results.
                    if none, use object's internal attribute specified during instantiation.
                top_sets : int, default = 5
                    number of rows to display of the performance summary table
                show_features : boolean, default=False
                    conditional controlling whether to print feature set for best validation
                    score.
                show_scores : int or none, default =None
                    display certain number of top features. if none, display nothing. if int, display
                    the specified number of features as a pandas DataFrame.
                marker_on : boolean, default=True
                    conditional controlling whether to display marker for each individual score.
                title_scale : float, default = 1.0
                    controls the scaling up (higher value) and scaling down (lower value) of the size of
                    the main chart title, the x_axis title and the y_axis title.
        """
        # load results summary if needed
        if isinstance(feature_selector_summary, str):
            feature_selector_summary = pd.read_csv(
                feature_selector_summary, index_col=0
            )
        elif isinstance(feature_selector_summary, pd.core.frame.DataFrame):
            feature_selector_summary = feature_selector_summary
        elif feature_selector_summary is None:
            try:
                feature_selector_summary = self.feature_selector_summary
            except AttributeError:
                raise AttributeError(
                    "no feature_selector_summary detected. either execute method feature_selector_suite or load from csv"
                )

        # load cv summary if needed
        if isinstance(cv_summary, str):
            cv_summary = pd.read_csv(cv_summary, index_col=0)
        elif isinstance(cv_summary, pd.core.frame.DataFrame):
            cv_summary = cv_summary
        elif cv_summary is None:
            try:
                cv_summary = self.cv_summary
            except AttributeError:
                raise AttributeError(
                    "no cv_summary detected. either execute method feature_selector_cross_val or load from csv"
                )

        for estimator in cv_summary["estimator"].unique():
            cv = cv_summary[
                (cv_summary["scoring"] == metric)
                & (cv_summary["estimator"] == estimator)
            ]

            total_features = feature_selector_summary.shape[0]
            iters = cv.shape[0]
            step = np.ceil(total_features / iters)

            cv.set_index(
                keys=np.arange(0, cv.shape[0] * step, step, dtype=int), inplace=True
            )

            if show_scores is not None:
                display(cv[:show_scores])

            # capture best iteration's feature drop count and performance score
            if metric in ["root_mean_squared_error", "root_mean_squared_log_error"]:
                sort_order = True
            else:
                sort_order = False

            num_dropped = cv.sort_values(["validation score"], ascending=sort_order)[
                :1
            ]["features dropped"].values[0]
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

            # create multi_line plot
            p = PrettierPlot()
            ax = p.make_canvas(
                title="{}\nBest validation {} = {}\nFeatures dropped = {}".format(
                    estimator, metric, score, num_dropped
                ),
                x_label="features removed",
                y_label=metric,
                y_shift=0.4 if len(metric) > 18 else 0.57,
                title_scale=title_scale,
            )

            p.pretty_multi_line(
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

    def create_cross_val_features_df(
        self, metric, cv_summary=None, feature_selector_summary=None
    ):
        """
        documentation:
            description:
                for each estimator, visualize the training and validation performance
                for each feature set.
            parameters:
                metric : string
                    metric to visualize.
                cv_summary : pandas DataFrame or str, default =None
                    pandas DataFrame, or str of csv file location, containing cross_validation results.
                    if none, use object's internal attribute specified during instantiation.
                feature_selector_summary : pandas DataFrame or str, default =None
                    pandas DataFrame, or str of csv file location, containing summary of feature_selector_suite results.
                    if none, use object's internal attribute specified during instantiation.
        """
        # load results summary if needed
        if isinstance(feature_selector_summary, str):
            feature_selector_summary = pd.read_csv(
                feature_selector_summary, index_col=0
            )
        elif isinstance(feature_selector_summary, pd.core.frame.DataFrame):
            feature_selector_summary = feature_selector_summary
        elif feature_selector_summary is None:
            try:
                feature_selector_summary = self.feature_selector_summary
            except AttributeError:
                raise AttributeError(
                    "no feature_selector_summary detected. either execute method feature_selector_suite or load from csv"
                )

        # load cv summary if needed
        if isinstance(cv_summary, str):
            cv_summary = pd.read_csv(cv_summary, index_col=0)
        elif isinstance(cv_summary, pd.core.frame.DataFrame):
            cv_summary = cv_summary
        elif cv_summary is None:
            try:
                cv_summary = self.cv_summary
            except AttributeError:
                raise AttributeError(
                    "no cv_summary detected. either execute method feature_selector_suite or load from csv"
                )

        # create empty DataFrame with feature names as index
        self.cross_val_features_df = pd.DataFrame(index=feature_selector_summary.index)

        # iterate through estimators
        for estimator in cv_summary["estimator"].unique():
            cv = cv_summary[
                (cv_summary["scoring"] == metric)
                & (cv_summary["estimator"] == estimator)
            ]
            cv = cv.reset_index(drop=True)

            # capture best iteration's feature drop count
            if metric in ["root_mean_squared_error", "root_mean_squared_log_error"]:
                sort_order = True
            else:
                sort_order = False

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

        # add counter and fill na_ns
        self.cross_val_features_df["count"] = self.cross_val_features_df.count(axis=1)
        self.cross_val_features_df = self.cross_val_features_df.fillna("")

        # add numerical index starting at 1
        self.cross_val_features_df = self.cross_val_features_df.reset_index()
        self.cross_val_features_df.index = np.arange(
            1, len(self.cross_val_features_df) + 1
        )
        self.cross_val_features_df = self.cross_val_features_df.rename(
            columns={"index": "feature"}
        )
        return self.cross_val_features_df

    def create_cross_val_features_dict(self, cross_val_features_df=None):
        """
        documentation:
            description:
                for each estimator, visualize the training and validation performance
                for each feature set.
            parameters:
                cross_val_features_df : pandas DataFrame, default =None
                    pandas DataFrame, or str of csv file location, containing summary of features used
                    in each estimator to achieve best validation score. if none, use object's internal
                    attribute specified during instantiation.
        """
        # load results summary if needed
        if isinstance(cross_val_features_df, str):
            cross_val_features_df = pd.read_csv(cross_val_features_df, index_col=0)
        elif isinstance(cross_val_features_df, pd.core.frame.DataFrame):
            cross_val_features_df = cross_val_features_df
        elif cross_val_features_df is None:
            try:
                cross_val_features_df = self.cross_val_features_df
            except AttributeError:
                raise AttributeError(
                    "no cross_val_features_df detected. either execute method create_cross_val_features_df or load from csv"
                )

        # create empty dict with feature names as index
        self.cross_val_features_dict = {}

        cross_val_features_df = cross_val_features_df.set_index("feature")
        cross_val_features_df = cross_val_features_df.drop("count", axis=1)

        # iterate through estimators
        for estimator in cross_val_features_df.columns:
            self.cross_val_features_dict[estimator] = cross_val_features_df[
                cross_val_features_df[estimator] == "x"
            ][estimator].index

        return self.cross_val_features_dict
