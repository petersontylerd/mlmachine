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
def objective(space, resultsFile, model, data, target, scoring, nFolds, nJobs, verbose):
    """
    Documentation:
        Description:
            Customizable objective function to be minimized in the Bayesian hyper-parameter optimization
            process.
        Parameters:
            space : dictionary
                Dictionary containg 'parameter : value distribution' key/value pairs. The key specifies the
                parameter of the model and optimization process draws trial values from the distribution.
            resultsFile : string
                File destination for results summary CSV.
            model : string
                The model to be fit.
            data : array
                Input dataset.
            target : array
                Input dataset labels.
            scoring : string (sklearn evaluation method)
                Evaluation method for scoring model performance. The following metrics are supported:
                    - "accuracy"
                    - "f1_macro"
                    - "f1_micro"
                    - "neg_mean_squared_error"
                    - "roc_auc"
                    - "root_mean_squared_error"
                    - "root_mean_squared_log_error"
                Please note that "root_mean_squared_log_error" is not implemented in sklearn. If "root_mean_squared_log_error" is specified, model
                is optimized using "neg_mean_squared_error" and then the square root is taken of the
                absolute value of the results, effectively creating the "root_mean_squared_log_error" score to be minimized.
            nFolds : int
                Number of folds for cross-validation.
            nJobs : int
                Number of works to deploy upon execution, if applicable.
            verbose : int
                Controls amount of information printed to console during fit.
        Returns:
            results : dictionary
                Dictionary containing details for each individual trial. Details include model type,
                iteration, parameter values, run time, and cross-validation summary statistics.
    """
    global ITERATION
    ITERATION += 1
    start = timer()

    # convert select float params to int
    for param in ["num_leaves", "subsample_for_bin", "min_child_samples"]:
        if param in space.keys():
            space[param] = int(space[param])

    # custom metric handling
    if scoring == "root_mean_squared_error":
        scoring = "neg_mean_squared_error"
        scoreTransform = "rmse"
    elif scoring == "root_mean_squared_log_error":
        scoring = "neg_mean_squared_log_error"
        scoreTransform = "rmsle"
    else:
        scoreTransform = scoring


    cv = model_selection.cross_val_score(
        estimator=eval("{0}(**{1})".format(model, space)),
        X=data,
        y=target,
        verbose=verbose,
        n_jobs=nJobs,
        cv=nFolds,
        scoring=scoring,
    )
    runTime = timer() - start

    # calculate loss based on scoring method
    if scoreTransform in ["accuracy","f1_micro","f1_macro","roc_auc"]:
        loss = 1 - cv.mean()

        meanScore = cv.mean()
        stdScore = cv.std()
        minScore = cv.min()
        maxScore = cv.max()
    # negative mean squared error
    elif scoreTransform in ["neg_mean_squared_error"]:
        loss = np.abs(cv.mean())

        meanScore = cv.mean()
        stdScore = cv.std()
        minScore = cv.min()
        maxScore = cv.max()
    # root mean squared error
    elif scoreTransform in ["rmse"]:
        cv = np.sqrt(np.abs(cv))
        loss = np.mean(cv)

        meanScore = cv.mean()
        stdScore = cv.std()
        minScore = cv.min()
        maxScore = cv.max()
    # root mean squared log error
    elif scoring in ["rmsle"]:
        cv = np.sqrt(np.abs(cv))
        loss = np.mean(cv)

        meanScore = cv.mean()
        stdScore = cv.std()
        minScore = cv.min()
        maxScore = cv.max()

    # export results to CSV
    outFile = resultsFile
    with open(outFile, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                ITERATION,
                model,
                scoring,
                loss,
                meanScore,
                stdScore,
                minScore,
                maxScore,
                runTime,
                STATUS_OK,
                space,
            ]
        )

    return {
        "iteration": ITERATION,
        "estimator": model,
        "scoring": scoring,
        "loss": loss,
        "meanScore": meanScore,
        "stdScore": stdScore,
        "minScore": minScore,
        "maxScore": maxScore,
        "trainTime": runTime,
        "status": STATUS_OK,
        "params": space,
    }


def execBayesOptimSearch(self, allSpace, data, target, scoring, columns=None, nFolds=3, nJobs=4, iters=50, verbose=0, resultsFile=None):
    """
    Documentation:
        Definition:
            Perform Bayesian hyper-parameter optimization across a set of models and parameter value
            distribution.
        Parameters:
            allSpace : dictionary of dictionaries
                Dictionary of nested dictionaries. Outer key is a model, and the corresponding value is
                a dictionary. Each nested dictionary contains 'parameter : value distribution' key/value
                pairs. The inner dictionary key specifies the parameter of the model to be tuned, and the
                value is a distribution of values from which trial values are drawn.
            model : string
                The model to be fit.
            data : array
                Input dataset.
            target : array
                Input dataset labels.
            scoring : string (sklearn evaluation method)
                Evaluation method for scoring model performance. Takes values "neg_mean_squared_error",
                 "accuracy", and "root_mean_squared_log_error". Please note that "root_mean_squared_log_error"
                 is not an actual sklearn evaluation method. If "root_mean_squared_log_error" is specified, model
                 is optimized using "neg_mean_squared_error" and then the square root is taken of the absolute value
                 of the results, effectively creating the "root_mean_squared_log_error" score to be minimized.
            columns : dict, default = None
                Dictionary containing str/list key/value pairs, where the str is the name of the estimator
                and the list contains string of column names. Enables utilization of different features sets for
                each estimator.
            nFolds : int, default = 3
                Number of folds for cross-validation.
            nJobs : int, default - 4
                Number of works to deploy upon execution, if applicable.
            verbose : int, default=0
                Controls amount of information printed to console during fit.
            resultsFile : string, default = None
                File destination for results summary CSV. If None, defaults to
                ./bayesOptimizationSummary_{data}_{time}.csv.
    """
    if resultsFile is None:
        resultsFile = "bayesOptimizationSummary_{}_{}.csv".format(scoring, strftime("%Y%m%d_%H%M%S", gmtime()))

    # add file header
    with open(resultsFile, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            [
                "iteration",
                "estimator",
                "scoring",
                "loss",
                "meanScore",
                "stdScore",
                "minScore",
                "maxScore",
                "trainTime",
                "status",
                "params",
            ]
        )

    # iterate through each model
    for estimator in allSpace.keys():
        global ITERATION
        ITERATION = 0

        # establish feature space for hyper-parameter search
        space = allSpace[estimator]

        # conditionally handle input data
        if isinstance(data, pd.core.frame.DataFrame):

            # filter input data based on estimator / column-subset pairs
            if columns is not None:
                try:
                    inputData = data[columns[estimator]]
                except KeyError:
                    inputData = data.copy()

                # use underlying numpy ndarray
                inputData = inputData.values

            elif columns is None:
                # use underlying numpy ndarray
                inputData = data.values
        elif isinstance(data, np.ndarray):
            # create a copy of the underlying data array
            inputData = data.copy()
        else:
            raise AttributeError("Input data set must be either a Pandas DataFrame or a numpy ndarray")

        # conditionally handle input target
        if isinstance(target, pd.core.series.Series):
            # use underlying numpy ndarray
            inputTarget = target.values
        elif isinstance(target, np.ndarray):
            # create a copy of the underlying data array
            inputTarget = target.copy()
        else:
            raise AttributeError("Input target must be either a Pandas Series or a numpy ndarray")

        # override default arguments with next estimator and the CV parameters
        objective.__defaults__ = (
            resultsFile,
            estimator,
            inputData,
            inputTarget,
            scoring,
            nFolds,
            nJobs,
            verbose,
        )

        # run optimization
        print("#" * 100)
        print("\nTuning {0}\n".format(estimator))
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
    Documentation:
        Description:
            Helper class for instantiating an input model type with a provided
            parameter set.
        Parameters:
            bayesOptimSummary : Pandas DataFrame
                Pandas DataFrame containing results from Bayesian Optimization process
                execution.
            estimator : string or sklearn API object
                Name of estimator to build. Needs the format of [submodule].[estimator].
            modelIter : int
                Numerical identifier for a specific parameter set used in a training
                iteration.
            nJobs : int, default = 4
                Number of works to use when training the model. This parameter will be
                ignored if the model does not have this parameter.
        Returns:
            model : model object
                Model instantiated using parameter set. Model possesses train, predict, fit
                and feature_importances methods.
    """

    def __init__(self, bayesOptimSummary, estimator, modelIter, nJobs=4):
        self.bayesOptimSummary=bayesOptimSummary
        self.estimator=estimator
        self.modelIter=modelIter
        self.nJobs=nJobs
        self.params = self.bayesOptimSummary[
            (self.bayesOptimSummary["estimator"] == self.estimator) & (self.bayesOptimSummary["iteration"] == self.modelIter)
        ]["params"].values[0]

        # turn string into dict
        self.params = ast.literal_eval(self.params)

        # convert estimator argument to sklearn API object if needed
        if isinstance(self.estimator, str):
            self.estimator = eval(self.estimator)

        # capture available model arguments and set probabily and n_jobs where applicable
        estimatorArgs = inspect.getfullargspec(self.estimator).args

        if "probability" in estimatorArgs:
            self.params['probability'] = True

        if "n_jobs" in estimatorArgs:
            self.params['n_jobs'] = self.nJobs

        # instantiate model
        self.model = self.estimator(**self.params)

    def train(self, XTrain, yTrain):
        self.model.fit(XTrain, yTrain)

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
    Documentation:
        Description:
            Helper class for instantiating an input model type.
        Parameters:
            estimator : sklearn model, as either an uncalled object or a string.
                Model to instantiate.
            params : dictionary, default = None
                Dictionary containing 'parameter : value' pairs. If no dictionary is provided,
                then an empty dictionary is created by default, which instantiates the model with
                its default parameter values.
            nJobs : int, default = 4
                Number of works to use when training the model. This parameter will be
                ignored if the model does not have this parameter.
        Returns:
            model : model object
                Model instantiated using parameter set. Model possesses train, predict, fit
                and feature_importances methods.
    """

    def __init__(self, estimator, params=None, nJobs=4, randomState=0):

        self.estimator=estimator
        self.params={} if params is None else params
        self.nJobs=nJobs
        self.randomState=randomState

        # convert estimator argument to sklearn API object if needed
        if isinstance(self.estimator, str):
            self.estimator = eval(self.estimator)

        # capture available model arguments and set probabily and n_jobs where applicable
        estimatorArgs = inspect.getfullargspec(self.estimator).args

        if "probability" in estimatorArgs:
            self.params['probability'] = True

        if "n_jobs" in estimatorArgs:
            self.params['n_jobs'] = self.nJobs

        if "random_state" in estimatorArgs:
            self.params['random_state'] = self.randomState

        # instantiate model
        self.model = self.estimator(**self.params)

    def train(self, XTrain, yTrain):
        self.model.fit(XTrain, yTrain)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def fit(self, x, y):
        return self.model.fit(x, y)

    def feature_importances(self, x, y):
        return self.model.fit(x, y).feature_importances_


def unpackBayesOptimSummary(self, bayesOptimSummary, estimator):
    """
    Documentation:
        Definition:
            Unpack Bayesian Optimization results summary data for a specified estimator
            into a Pandas DataFrame where there is one column for each parameter.
        Parameters
            bayesOptimSummary : Pandas DataFrame
                Pandas DataFrame containing results from Bayesian Optimization process
                execution.
            estimator : string
                Name of estimator to build. Needs the format of [submodule].[estimator].
        Returns:
            estimatorParamSummary : Pandas DataFrame
                Pandas DataFrame where each row is a record of the parameters used and the
                loss recorded in an iteration.
    """

    estimatorDf = bayesOptimSummary[bayesOptimSummary["estimator"] == estimator].reset_index(drop=True)

    # Create a new dataframe for storing parameters
    estimatorSummary = pd.DataFrame(
        columns=list(ast.literal_eval(estimatorDf.loc[0, "params"]).keys()),
        index=list(range(len(estimatorDf))),
    )

    # Add the results with each parameter a different column
    for i, params in enumerate(estimatorDf["params"]):
        estimatorSummary.loc[i, :] = list(ast.literal_eval(params).values())

    # add columns for loss and iter number
    estimatorSummary["iterLoss"] = estimatorDf["loss"]
    estimatorSummary["iteration"] = estimatorDf["iteration"]

    return estimatorSummary


def modelLossPlot(self, bayesOptimSummary, estimator, chartProp=15, trimOutliers=True, outlierControl=1.5, titleScale=0.7):
    """
    Documentation:
        Definition:
            Visualize how the Bayesian Optimization loss change over time across all iterations.
            Extremely poor results are removed from visualized dataset by two filters.
                1) Loss values worse than [loss mean + (2 x loss std)]
                2) Los values worse than [median * outliersControl]. 'outlierControl' is a parameter
                   that can be set during function execution.
        Parameters
            bayesOptimSummary : Pandas DataFrame
                Pandas DataFrame containing results from Bayesian Optimization process
                execution.
            estimator : string
                Name of estimator to build. Needs the format of [submodule].[estimator].
            chartProp : float, default = 15
                Control chart proportions. Higher values scale up size of chart objects, lower
                values scale down size of chart objects.
            trimOutliers : boolean, default = True
                This removes extremely high (poor) results by trimming values that observations where
                the loss is greater than 2 standard deviations away from the mean.
            outlierControl : float: default = 1.5
                Controls enforcement of outlier trimming. Value is multiplied by median, and the resulting
                product is the cap placed on loss values. Values higher than this cap will be excluded.
                Lower values of outlierControl apply more extreme filtering of loss values.
            titleScale : float, default = 0.7
                Controls the scaling up (higher value) and scaling down (lower value) of the size of
                the main chart title, the x-axis title and the y-axis title.
    """
    estimatorSummary = self.unpackBayesOptimSummary(bayesOptimSummary=bayesOptimSummary, estimator=estimator)
    if trimOutliers:
        mean = estimatorSummary['iterLoss'].mean()
        median = estimatorSummary['iterLoss'].median()
        std = estimatorSummary['iterLoss'].std()
        cap = mean + (2.0*std)
        estimatorSummary = estimatorSummary[(estimatorSummary['iterLoss'] < cap) & (estimatorSummary['iterLoss'] < outlierControl * median)]

    # create regression plot
    p = PrettierPlot(chartProp=chartProp)
    ax = p.makeCanvas(
        title="Regression plot\n* {}".format(estimator),
        yShift=0.8,
        position=111,
        titleScale=titleScale,
    )
    p.prettyRegPlot(x="iteration",
        y="iterLoss",
        data=estimatorSummary,
        yUnits="ffff",
        ax=ax
    )
    plt.show()


def modelParamPlot(self, bayesOptimSummary, estimator, allSpace, nIter, chartProp = 17, titleScale=0.7):
    """
    Documentation:
        Definition:
            Visualize hyper-parameter optimization over the iterations. Compares theoretical
            distribution to the distribution of values that were actually chosen. For parameters
            with a numeric range of values, this function also visualizes how the parameter
            value changes over time.
        Parameters
            bayesOptimSummary : Pandas DataFrame
                Pandas DataFrame containing results from Bayesian Optimization process
                execution.
            estimator : string
                Name of estimator to build. Needs the format of [submodule].[estimator].
            allSpace : dictionary of dictionaries
                Dictionary of nested dictionaries. Outer key is a model, and the corresponding value is
                a dictionary. Each nested dictionary contains 'parameter : value distribution' key/value
                pairs. The inner dictionary key specifies the parameter of the model to be tuned, and the
                value is a distribution of values from which trial values are drawn.
            nIter : int
                Number of iterations to draw from theoretical distribution in order to visualize the
                theoretical distribution. Higher number leader to more robust distribution but can take
                considerably longer to create.
            chartProp : float, default = 10
                Controls proportions of visualizations. Larger values scale visual up in size, smaller values
                scale visual down in size.
            titleScale : float, default = 0.7
                Controls the scaling up (higher value) and scaling down (lower value) of the size of
                the main chart title, the x-axis title and the y-axis title.
    """
    estimatorSummary = self.unpackBayesOptimSummary(bayesOptimSummary = bayesOptimSummary, estimator = estimator)

    # return space belonging to estimator
    estimatorSpace = allSpace[estimator]

    print('*' * 100)
    print('* {}'.format(estimator))
    print('*' * 100)

    # iterate through each parameter
    for param in estimatorSpace.keys():
        # create data to represent theoretical distribution
        theoreticalDist = []
        for _ in range(nIter):
            theoreticalDist.append(sample(estimatorSpace)[param])

        # clean up
        theoreticalDist = ["None" if v is None else v for v in theoreticalDist]
        theoreticalDist = np.array(theoreticalDist)

        # clean up
        actualDist = estimatorSummary[param].tolist()
        actualDist = ["None" if v is None else v for v in actualDist]
        actualDist = np.array(actualDist)

        actualIterDf = estimatorSummary[["iteration",param]]

        # plot distributions for categorical params
        if any(isinstance(d, str) for d in theoreticalDist):

            p = PrettierPlot(chartProp=chartProp)
            # theoretical plot
            uniqueVals, uniqueCounts = np.unique(
                theoreticalDist, return_counts=True
            )
            ax = p.makeCanvas(
                title="Theoretical plot\n* {}".format(param),
                yShift=0.8,
                position=121,
                titleScale=titleScale,
            )
            p.prettyBarV(
                x=uniqueVals,
                counts=uniqueCounts,
                labelRotate=90 if len(uniqueVals) >= 4 else 0,
                color=style.styleGrey,
                yUnits="f",
                ax=ax,
            )

            # actual plot
            uniqueVals, uniqueCounts = np.unique(actualDist, return_counts=True)
            if param == 'loss':
                print(uniqueVals)
                print(uniqueCounts)

            ax = p.makeCanvas(
                title="Actual plot\n* {}".format(param),
                yShift=0.8,
                position=122,
                titleScale=titleScale,
            )
            p.prettyBarV(
                x=uniqueVals,
                counts=uniqueCounts,
                labelRotate=90 if len(uniqueVals) >= 4 else 0,
                color=style.styleBlue,
                yUnits="f",
                ax=ax,
            )

            plt.show()

        # plot distributions for numeric params
        else:
            # using dictionary to convert specific columns
            convertDict = {"iteration": int, param: float}

            actualIterDf = actualIterDf.astype(convertDict)

            p = PrettierPlot(chartProp=chartProp)
            # p = PrettierPlot(chartProp=8, plotOrientation="square")
            ax = p.makeCanvas(
                title="Actual vs. theoretical plot\n* {}".format(param),
                yShift=0.8,
                position=121,
                titleScale=titleScale,
            )
            p.prettyKdePlot(
                theoreticalDist,
                color=style.styleGrey,
                yUnits="ppp",
                xUnits="fff" if np.max(theoreticalDist) <= 5.0 else "ff",
                ax=ax,
            )
            p.prettyKdePlot(
                actualDist,
                color=style.styleBlue,
                yUnits="ppp",
                xUnits="fff" if np.max(actualDist) <= 5.0 else "ff",
                ax=ax,
            )

            ## create custom legend
            # create labels
            labelColor = {}
            legendLabels = ['Theoretical','Actual']
            colorList = [style.styleGrey, style.styleBlue]
            for ix, i in enumerate(legendLabels):
                labelColor[i] = colorList[ix]

            # create patches
            patches = [Patch(color=v, label=k) for k, v in labelColor.items()]

            # draw legend
            leg = plt.legend(
                handles=patches,
                fontsize=0.8 * chartProp,
                loc="upper right",
                markerscale=0.3 * chartProp,
                ncol=1,
                bbox_to_anchor=(1.05, 1.1),
            )

            # label font color
            for text in leg.get_texts():
                plt.setp(text, color="Grey")


            ax = p.makeCanvas(
                title="Regression plot\n* {}".format(estimator),
                yShift=0.8,
                position=122,
                titleScale=titleScale,
            )
            p.prettyRegPlot(
                x="iteration", y=param, data=actualIterDf, yUnits="fff", ax=ax
            )
            plt.show()


def samplePlot(self, sampleSpace, nIter, chartProp=15):
    """
    Documentation:
        Definition:
            Visualizes a single hyperopt theoretical distribution. Useful for helping to determine a
            distribution to use when setting up hyperopt distribution objects for actual parameter
            tuning.
        Parameters
            sampleSpace : dictionary
                Dictionary of 'param name : hyperopt distribution object' key/value pairs. The name can
                be arbitrarily chosen, and the value is a defined hyperopt distribution.
            nIter : int
                Number of iterations to draw from theoretical distribution in order to visualize the
                theoretical distribution. Higher number leader to more robust distribution but can take
                considerably longer to create.
    """

    # iterate through each parameter
    for param in sampleSpace.keys():
        # create data to represent theoretical distribution
        theoreticalDist = []
        for _ in range(nIter):
            theoreticalDist.append(sample(sampleSpace)[param])
        theoreticalDist = np.array(theoreticalDist)

        p = PrettierPlot(chartProp=chartProp)
        ax = p.makeCanvas(
            title="Actual vs. theoretical plot\n* {}".format(param),
            yShift=0.8,
            position=111,
        )
        p.prettyKdePlot(
            theoreticalDist,
            color=style.styleGrey,
            yUnits="p",
            xUnits="fff" if np.max(theoreticalDist) <= 5.0 else "ff",
            ax=ax,
        )