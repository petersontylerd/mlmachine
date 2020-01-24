import ast
import csv
import inspect
import sys
import time
from time import gmtime, strftime
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import pandas as pd

from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from hyperopt.pyll.stochastic import sample

import sklearn.model_selection as model_selection

import sklearn.decomposition as decomposition
import sklearn.discriminant_analysis as discriminant_analysis
import sklearn.ensemble as ensemble
import sklearn.gaussian_process as gaussian_process
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


# set optimization parameters
def objective(space, results_file, model, data, target, scoring, n_folds, n_jobs, verbose):
    """
    documentation:
        description:
            customizable objective function to be minimized in the bayesian hyper_parameter optimization
            process.
        parameters:
            space : dictionary
                dictionary containg 'parameter : value distribution' key/value pairs. the key specifies the
                parameter of the model and optimization process draws trial values from the distribution.
            results_file : string
                file destination for results summary csv.
            model : string
                the model to be fit.
            data : array
                input dataset.
            target : array
                input dataset labels.
            scoring : string (sklearn evaluation method)
                evaluation method for scoring model performance. the following metrics are supported:
                    - "accuracy"
                    - "f1_macro"
                    - "f1_micro"
                    - "neg_mean_squared_error"
                    - "roc_auc"
                    - "root_mean_squared_error"
                    - "root_mean_squared_log_error"
                please note that "root_mean_squared_log_error" is not implemented in sklearn. if "root_mean_squared_log_error" is specified, model
                is optimized using "neg_mean_squared_error" and then the square root is taken of the
                absolute value of the results, effectively creating the "root_mean_squared_log_error" score to be minimized.
            n_folds : int
                number of folds for cross_validation.
            n_jobs : int
                number of works to deploy upon execution, if applicable.
            verbose : int
                controls amount of information printed to console during fit.
        returns:
            results : dictionary
                dictionary containing details for each individual trial. details include model type,
                iteration, parameter values, run time, and cross_validation summary statistics.
    """
    global iteration
    iteration += 1
    start = timer()

    # convert select float params to int
    for param in ["num_leaves", "subsample_for_bin", "min_child_samples"]:
        if param in space.keys():
            space[param] = int(space[param])

    # custom metric handling
    if scoring == "root_mean_squared_error":
        scoring = "neg_mean_squared_error"
        score_transform = "rmse"
    elif scoring == "root_mean_squared_log_error":
        scoring = "neg_mean_squared_log_error"
        score_transform = "rmsle"
    else:
        score_transform = scoring

    cv = model_selection.cross_val_score(
        estimator=eval("{0}(**{1})".format(model, space)),
        X=data,
        y=target,
        verbose=verbose,
        n_jobs=n_jobs,
        cv=n_folds,
        scoring=scoring,
    )
    run_time = timer() - start

    # calculate loss based on scoring method
    if score_transform in ["accuracy", "f1_micro", "f1_macro", "roc_auc"]:
        loss = 1 - cv.mean()

        mean_score = cv.mean()
        std_score = cv.std()
        min_score = cv.min()
        max_score = cv.max()
    # negative mean squared error
    elif score_transform in ["neg_mean_squared_error"]:
        loss = np.abs(cv.mean())

        mean_score = cv.mean()
        std_score = cv.std()
        min_score = cv.min()
        max_score = cv.max()
    # root mean squared error
    elif score_transform in ["rmse"]:
        cv = np.sqrt(np.abs(cv))
        loss = np.mean(cv)

        mean_score = cv.mean()
        std_score = cv.std()
        min_score = cv.min()
        max_score = cv.max()
    # root mean squared log error
    elif scoring in ["rmsle"]:
        cv = np.sqrt(np.abs(cv))
        loss = np.mean(cv)

        mean_score = cv.mean()
        std_score = cv.std()
        min_score = cv.min()
        max_score = cv.max()

    # export results to csv
    out_file = results_file
    with open(out_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                iteration,
                model,
                scoring,
                loss,
                mean_score,
                std_score,
                min_score,
                max_score,
                run_time,
                STATUS_OK,
                space,
            ]
        )

    return {
        "iteration": iteration,
        "estimator": model,
        "scoring": scoring,
        "loss": loss,
        "mean_score": mean_score,
        "std_score": std_score,
        "min_score": min_score,
        "max_score": max_score,
        "train_time": run_time,
        "status": STATUS_OK,
        "params": space,
    }


def exec_bayes_optim_search(self, all_space, data, target, scoring, columns=None, n_folds=3,
                        n_jobs=4, iters=50, verbose=0, results_file=None,):
    """
    documentation:
        definition:
            perform bayesian hyper_parameter optimization across a set of models and parameter value
            distribution.
        parameters:
            all_space : dictionary of dictionaries
                dictionary of nested dictionaries. outer key is a model, and the corresponding value is
                a dictionary. each nested dictionary contains 'parameter : value distribution' key/value
                pairs. the inner dictionary key specifies the parameter of the model to be tuned, and the
                value is a distribution of values from which trial values are drawn.
            model : string
                the model to be fit.
            data : array
                input dataset.
            target : array
                input dataset labels.
            scoring : string (sklearn evaluation method)
                evaluation method for scoring model performance. takes values "neg_mean_squared_error",
                 "accuracy", and "root_mean_squared_log_error". please note that "root_mean_squared_log_error"
                 is not an actual sklearn evaluation method. if "root_mean_squared_log_error" is specified, model
                 is optimized using "neg_mean_squared_error" and then the square root is taken of the absolute value
                 of the results, effectively creating the "root_mean_squared_log_error" score to be minimized.
            columns : dict, default=None
                dictionary containing str/list key/value pairs, where the str is the name of the estimator
                and the list contains string of column names. enables utilization of different features sets for
                each estimator.
            n_folds : int, default = 3
                number of folds for cross_validation.
            n_jobs : int, default - 4
                number of works to deploy upon execution, if applicable.
            verbose : int, default=0
                controls amount of information printed to console during fit.
            results_file : string, default=None
                file destination for results summary csv. if none, defaults to
                ./bayes_optimization_summary_{data}_{time}.csv.
    """
    if results_file is None:
        results_file = "bayes_optimization_summary_{}_{}.csv".format(
            scoring, strftime("%y%m%d%H%M", gmtime())
        )

    # add file header
    with open(results_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            [
                "iteration",
                "estimator",
                "scoring",
                "loss",
                "mean_score",
                "std_score",
                "min_score",
                "max_score",
                "train_time",
                "status",
                "params",
            ]
        )

    # iterate through each model
    for estimator in all_space.keys():
        global iteration
        iteration = 0

        # establish feature space for hyper_parameter search
        space = all_space[estimator]

        # conditionally handle input data
        if isinstance(data, pd.core.frame.DataFrame):

            # filter input data based on estimator / column_subset pairs
            if columns is not None:
                try:
                    input_data = data[columns[estimator]]
                except KeyError:
                    input_data = data.copy()

                # use underlying numpy ndarray
                input_data = input_data.values

            elif columns is None:
                # use underlying numpy ndarray
                input_data = data.values
        elif isinstance(data, np.ndarray):
            # create a copy of the underlying data array
            input_data = data.copy()
        else:
            raise AttributeError(
                "input data set must be either a pandas DataFrame or a numpy ndarray"
            )

        # conditionally handle input target
        if isinstance(target, pd.core.series.Series):
            # use underlying numpy ndarray
            input_target = target.values
        elif isinstance(target, np.ndarray):
            # create a copy of the underlying data array
            input_target = target.copy()
        else:
            raise AttributeError(
                "input target must be either a Pandas Series or a numpy ndarray"
            )

        # override default arguments with next estimator and the cv parameters
        objective.__defaults__ = (
            results_file,
            estimator,
            input_data,
            input_target,
            scoring,
            n_folds,
            n_jobs,
            verbose,
        )

        # run optimization
        print("#" * 100)
        print("\n_tuning {0}\n".format(estimator))
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=iters,
            trials=Trials(),
            verbose=verbose,
        )


class BayesOptimModelBuilder:
    """
    documentation:
        description:
            helper class for instantiating an input model type with a provided
            parameter set.
        parameters:
            bayes_optim_summary : pandas DataFrame
                pandas DataFrame containing results from bayesian optimization process
                execution.
            estimator : string or sklearn api object
                name of estimator to build. needs the format of [submodule].[estimator].
            model_iter : int
                numberal identifier for a specific parameter set used in a training
                iteration.
            n_jobs : int, default = 4
                number of works to use when training the model. this parameter will be
                ignored if the model does not have this parameter.
        returns:
            model : model object
                model instantiated using parameter set. model possesses train, predict, fit
                and feature_importances methods.
    """

    def __init__(self, bayes_optim_summary, estimator, model_iter, n_jobs=4):
        self.bayes_optim_summary = bayes_optim_summary
        self.estimator = estimator
        self.model_iter = model_iter
        self.n_jobs = n_jobs
        self.params = self.bayes_optim_summary[
            (self.bayes_optim_summary["estimator"] == self.estimator)
            & (self.bayes_optim_summary["iteration"] == self.model_iter)
        ]["params"].values[0]

        # turn string into dict
        self.params = ast.literal_eval(self.params)

        # convert estimator argument to sklearn api object if needed
        if isinstance(self.estimator, str):
            self.estimator = eval(self.estimator)

        # capture available model arguments and set probabily and n_jobs where applicable
        estimator_args = inspect.getfullargspec(self.estimator).args

        if "probability" in estimator_args:
            self.params["probability"] = True

        if "n_jobs" in estimator_args:
            self.params["n_jobs"] = self.n_jobs

        # instantiate model
        self.model = self.estimator(**self.params)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def fit(self, x, y):
        return self.model.fit(x, y)

    def feature_importances(self, x, y):
        return self.model.fit(x, y).feature_importances_


class BasicModelBuilder:
    """
    documentation:
        description:
            helper class for instantiating an input model type.
        parameters:
            estimator : sklearn model, as either an uncalled object or a string.
                model to instantiate.
            params : dictionary, default=None
                dictionary containing 'parameter : value' pairs. if no dictionary is provided,
                then an empty dictionary is created by default, which instantiates the model with
                its default parameter values.
            n_jobs : int, default = 4
                number of works to use when training the model. this parameter will be
                ignored if the model does not have this parameter.
        returns:
            model : model object
                model instantiated using parameter set. model possesses train, predict, fit
                and feature_importances methods.
    """

    def __init__(self, estimator, params=None, n_jobs=4, random_state=0):

        self.estimator = estimator
        self.params = {} if params is None else params
        self.n_jobs = n_jobs
        self.random_state = random_state

        # convert estimator argument to sklearn api object if needed
        if isinstance(self.estimator, str):
            self.estimator = eval(self.estimator)

        # capture available model arguments and set probabily and n_jobs where applicable
        estimator_args = inspect.getfullargspec(self.estimator).args

        if "probability" in estimator_args:
            self.params["probability"] = True

        if "n_jobs" in estimator_args:
            self.params["n_jobs"] = self.n_jobs

        if "random_state" in estimator_args:
            self.params["random_state"] = self.random_state

        # instantiate model
        self.model = self.estimator(**self.params)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def fit(self, x, y):
        return self.model.fit(x, y)

    def feature_importances(self, x, y):
        return self.model.fit(x, y).feature_importances_


def unpack_bayes_optim_summary(self, bayes_optim_summary, estimator):
    """
    documentation:
        definition:
            unpack bayesian optimization results summary data for a specified estimator
            into a pandas DataFrame where there is one column for each parameter.
        parameters
            bayes_optim_summary : pandas DataFrame
                pandas DataFrame containing results from bayesian optimization process
                execution.
            estimator : string
                name of estimator to build. needs the format of [submodule].[estimator].
        returns:
            estimator_param_summary : pandas DataFrame
                pandas DataFrame where each row is a record of the parameters used and the
                loss recorded in an iteration.
    """

    estimator_df = bayes_optim_summary[
        bayes_optim_summary["estimator"] == estimator
    ].reset_index(drop=True)

    # create a new dataframe for storing parameters
    estimator_summary = pd.DataFrame(
        columns=list(ast.literal_eval(estimator_df.loc[0, "params"]).keys()),
        index=list(range(len(estimator_df))),
    )

    # add the results with each parameter a different column
    for i, params in enumerate(estimator_df["params"]):
        estimator_summary.loc[i, :] = list(ast.literal_eval(params).values())

    # add columns for loss and iter number
    estimator_summary["iter_loss"] = estimator_df["loss"]
    estimator_summary["iteration"] = estimator_df["iteration"]

    return estimator_summary


def model_loss_plot(
    self,
    bayes_optim_summary,
    estimator,
    chart_scale=15,
    trim_outliers=True,
    outlier_control=1.5,
    title_scale=0.7,
):
    """
    documentation:
        definition:
            visualize how the bayesian optimization loss change over time across all iterations.
            extremely poor results are removed from visualized dataset by two filters.
                1) loss values worse than [loss mean + (2 x loss std)]
                2) los values worse than [median * outliers_control]. 'outlier_control' is a parameter
                   that can be set during function execution.
        parameters
            bayes_optim_summary : pandas DataFrame
                pandas DataFrame containing results from bayesian optimization process
                execution.
            estimator : string
                name of estimator to build. needs the format of [submodule].[estimator].
            chart_scale : float, default = 15
                control chart proportions. higher values scale up size of chart objects, lower
                values scale down size of chart objects.
            trim_outliers : bool, default=True
                this removes extremely high (poor) results by trimming values that observations where
                the loss is greater than 2 standard deviations away from the mean.
            outlier_control : float: default = 1.5
                controls enforcement of outlier trimming. value is multiplied by median, and the resulting
                product is the cap placed on loss values. values higher than this cap will be excluded.
                lower values of outlier_control apply more extreme filtering of loss values.
            title_scale : float, default = 0.7
                controls the scaling up (higher value) and scaling down (lower value) of the size of
                the main chart title, the x_axis title and the y_axis title.
    """
    estimator_summary = self.unpack_bayes_optim_summary(
        bayes_optim_summary=bayes_optim_summary, estimator=estimator
    )
    if trim_outliers:
        mean = estimator_summary["iter_loss"].mean()
        median = estimator_summary["iter_loss"].median()
        std = estimator_summary["iter_loss"].std()
        cap = mean + (2.0 * std)
        estimator_summary = estimator_summary[
            (estimator_summary["iter_loss"] < cap)
            & (estimator_summary["iter_loss"] < outlier_control * median)
        ]

    # create regression plot
    p = PrettierPlot(chart_scale=chart_scale)
    ax = p.make_canvas(
        title="regression plot\n* {}".format(estimator),
        y_shift=0.8,
        position=111,
        title_scale=title_scale,
    )
    p.reg_plot(
        x="iteration", y="iter_loss", data=estimator_summary, y_units="ffff", ax=ax
    )
    plt.show()


def model_param_plot(
    self,
    bayes_optim_summary,
    estimator,
    all_space,
    n_iter,
    chart_scale=17,
    title_scale=0.7,
):
    """
    documentation:
        definition:
            visualize hyper_parameter optimization over the iterations. compares theoretical
            distribution to the distribution of values that were actually chosen. for parameters
            with a number range of values, this function also visualizes how the parameter
            value changes over time.
        parameters
            bayes_optim_summary : pandas DataFrame
                pandas DataFrame containing results from bayesian optimization process
                execution.
            estimator : string
                name of estimator to build. needs the format of [submodule].[estimator].
            all_space : dictionary of dictionaries
                dictionary of nested dictionaries. outer key is a model, and the corresponding value is
                a dictionary. each nested dictionary contains 'parameter : value distribution' key/value
                pairs. the inner dictionary key specifies the parameter of the model to be tuned, and the
                value is a distribution of values from which trial values are drawn.
            n_iter : int
                number of iterations to draw from theoretical distribution in order to visualize the
                theoretical distribution. higher number leader to more robust distribution but can take
                considerably longer to create.
            chart_scale : float, default = 10
                controls proportions of visualizations. larger values scale visual up in size, smaller values
                scale visual down in size.
            title_scale : float, default = 0.7
                controls the scaling up (higher value) and scaling down (lower value) of the size of
                the main chart title, the x_axis title and the y_axis title.
    """
    estimator_summary = self.unpack_bayes_optim_summary(
        bayes_optim_summary=bayes_optim_summary, estimator=estimator
    )

    # return space belonging to estimator
    estimator_space = all_space[estimator]

    print("*" * 100)
    print("* {}".format(estimator))
    print("*" * 100)

    # iterate through each parameter
    for param in estimator_space.keys():
        # create data to represent theoretical distribution
        theoretical_dist = []
        for _ in range(n_iter):
            theoretical_dist.append(sample(estimator_space)[param])

        # clean up
        theoretical_dist = ["none" if v is None else v for v in theoretical_dist]
        theoretical_dist = np.array(theoretical_dist)

        # clean up
        actual_dist = estimator_summary[param].tolist()
        actual_dist = ["none" if v is None else v for v in actual_dist]
        actual_dist = np.array(actual_dist)

        actual_iter_df = estimator_summary[["iteration", param]]

        # plot distributions for object params
        if any(isinstance(d, str) for d in theoretical_dist):

            p = PrettierPlot(chart_scale=chart_scale)
            # theoretical plot
            unique_vals, unique_counts = np.unique(theoretical_dist, return_counts=True)
            ax = p.make_canvas(
                title="theoretical plot\n* {}".format(param),
                y_shift=0.8,
                position=121,
                title_scale=title_scale,
            )
            p.bar_v(
                x=unique_vals,
                counts=unique_counts,
                label_rotate=90 if len(unique_vals) >= 4 else 0,
                color=style.style_grey,
                y_units="f",
                ax=ax,
            )

            # actual plot
            unique_vals, unique_counts = np.unique(actual_dist, return_counts=True)
            if param == "loss":
                print(unique_vals)
                print(unique_counts)

            ax = p.make_canvas(
                title="actual plot\n* {}".format(param),
                y_shift=0.8,
                position=122,
                title_scale=title_scale,
            )
            p.bar_v(
                x=unique_vals,
                counts=unique_counts,
                label_rotate=90 if len(unique_vals) >= 4 else 0,
                color=style.style_blue,
                y_units="f",
                ax=ax,
            )

            plt.show()

        # plot distributions for number params
        else:
            # using dictionary to convert specific columns
            convert_dict = {"iteration": int, param: float}

            actual_iter_df = actual_iter_df.astype(convert_dict)

            p = PrettierPlot(chart_scale=chart_scale)
            # p = PrettierPlot(chart_scale=8, plot_orientation="square")
            ax = p.make_canvas(
                title="actual vs. theoretical plot\n* {}".format(param),
                y_shift=0.8,
                position=121,
                title_scale=title_scale,
            )
            p.kde_plot(
                theoretical_dist,
                color=style.style_grey,
                y_units="ppp",
                x_units="fff" if np.max(theoretical_dist) <= 5.0 else "ff",
                ax=ax,
            )
            p.kde_plot(
                actual_dist,
                color=style.style_blue,
                y_units="ppp",
                x_units="fff" if np.max(actual_dist) <= 5.0 else "ff",
                ax=ax,
            )

            ## create custom legend
            # create labels
            label_color = {}
            legend_labels = ["theoretical", "actual"]
            color_list = [style.style_grey, style.style_blue]
            for ix, i in enumerate(legend_labels):
                label_color[i] = color_list[ix]

            # create Patches
            Patches = [Patch(color=v, label=k, alpha=alpha) for k, v in label_color.items()]

            # draw legend
            leg = plt.legend(
                handles=Patches,
                fontsize=0.8 * chart_scale,
                loc="upper right",
                markerscale=0.3 * chart_scale,
                ncol=1,
                bbox_to_anchor=(1.05, 1.1),
            )

            # label font color
            for text in leg.get_texts():
                plt.setp(text, color="grey")

            ax = p.make_canvas(
                title="regression plot\n* {}".format(estimator),
                y_shift=0.8,
                position=122,
                title_scale=title_scale,
            )
            p.reg_plot(
                x="iteration", y=param, data=actual_iter_df, y_units="fff", ax=ax
            )
            plt.show()


def sample_plot(self, sample_space, n_iter, chart_scale=15):
    """
    documentation:
        definition:
            visualizes a single hyperopt theoretical distribution. useful for helping to determine a
            distribution to use when setting up hyperopt distribution objects for actual parameter
            tuning.
        parameters
            sample_space : dictionary
                dictionary of 'param name : hyperopt distribution object' key/value pairs. the name can
                be arbitrarily chosen, and the value is a defined hyperopt distribution.
            n_iter : int
                number of iterations to draw from theoretical distribution in order to visualize the
                theoretical distribution. higher number leader to more robust distribution but can take
                considerably longer to create.
    """

    # iterate through each parameter
    for param in sample_space.keys():
        # create data to represent theoretical distribution
        theoretical_dist = []
        for _ in range(n_iter):
            theoretical_dist.append(sample(sample_space)[param])
        theoretical_dist = np.array(theoretical_dist)

        p = PrettierPlot(chart_scale=chart_scale)
        ax = p.make_canvas(
            title="actual vs. theoretical plot\n* {}".format(param),
            y_shift=0.8,
            position=111,
        )
        p.kde_plot(
            theoretical_dist,
            color=style.style_grey,
            y_units="p",
            x_units="fff" if np.max(theoretical_dist) <= 5.0 else "ff",
            ax=ax,
        )
