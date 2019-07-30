import ast
import csv
import sys
import time
from timeit import default_timer as timer

import time

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
def objective(space, resultsDir, model, X, y, scoring, n_folds, n_jobs, verbose):
    """
    Documentation:
        Description:
            Customizable objective function to be minimized in the Bayesian hyper-parameter optimization
            process.
        Parameters:
            space : dictionary
                Dictionary containg 'parameter : value distribution' key/value pairs. The key specifies the 
                parameter of the model and optimization process draws trial values from the distribution.
            resultsDir : string
                File destination for results summary CSV.
            model : string
                The model to be fit.
            X : array
                Input dataset.
            y : array
                Input dataset labels.
            scoring : string (sklearn evaluation method)
                Evaluation method for scoring model performance. The following metrics are supported:
                    - "neg_mean_squared_error"
                    - "f1_macro"
                    - "f1_micro"
                    - "accuracy"
                    - "rmsle"
                Please note that "rmsle" is not implemented in sklearn. If "rmsle" is specified, model 
                is optimized using "neg_mean_squared_error" and then the square root is taken of the 
                absolute value of the results, effectively creating the "rmsle" score to be minimized.
            n_folds : int
                Number of folds for cross-validation.
            n_jobs : int
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

    cv = model_selection.cross_val_score(
        estimator=eval("{0}(**{1})".format(model, space)),
        X=X,
        y=y,
        verbose=verbose,
        n_jobs=n_jobs,
        cv=n_folds,
        scoring="neg_mean_squared_error" if scoring == "rmsle" else scoring,
    )
    run_time = timer() - start

    # calculate loss based on scoring method
    if scoring == "accuracy":
        loss = 1 - cv.mean()
    elif scoring == "f1_macro":
        loss = 1 - cv.mean()
    elif scoring == "f1_micro":
        loss = 1 - cv.mean()
    elif scoring == "neg_mean_squared_error":
        loss = abs(cv.mean())
    elif scoring == "rmsle":
        cv = np.sqrt(abs(cv))
        loss = np.mean(cv)

    # export results to CSV
    outFile = resultsDir
    with open(outFile, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                ITERATION,
                model,
                space,
                loss,
                cv.min(),
                cv.mean(),
                cv.max(),
                cv.std(),
                run_time,
                STATUS_OK,
            ]
        )

    return {
        "iteration": ITERATION,
        "estimator": model,
        "params": space,
        "loss": loss,
        "min": cv.min(),
        "mean": cv.mean(),
        "max": cv.max(),
        "std": cv.std(),
        "train_time": run_time,
        "status": STATUS_OK,
    }


def execBayesOptimSearch(self, allSpace, resultsDir, X, y, scoring, n_folds, n_jobs, iters, verbose):
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
            resultsDir : string
                File destination for results summary CSV.
            model : string
                The model to be fit.
            X : array
                Input dataset.
            y : array
                Input dataset labels.
            scoring : string (sklearn evaluation method)
                Evaluation method for scoring model performance. Takes values "neg_mean_squared_error",
                 "accuracy", and "rmsle". Please note that "rmsle" is not an actual sklearn evaluation
                 method. If "rmsle" is specified, model is optimized using "neg_mean_squared_error" and
                 then the square root is taken of the absolute value of the results, effectively creating
                 the "rmsle" score to be minimized.
            n_folds : int
                Number of folds for cross-validation.
            n_jobs : int
                Number of works to deploy upon execution, if applicable.
            verbose : int
                Controls amount of information printed to console during fit.
    """
    # add file header
    with open(resultsDir, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            [
                "iteration",
                "estimator",
                "params",
                "loss",
                "min",
                "mean",
                "max",
                "std",
                "train_time",
                "status",
            ]
        )

    # iterate through each model 
    for estimator in allSpace.keys():
        global ITERATION
        ITERATION = 0
        
        # establish feature space for hyper-parameter search
        space = allSpace[estimator]
        
        # override default arguments with next estimator and the CV parameters
        objective.__defaults__ = (
            resultsDir,
            estimator,
            X,
            y,
            scoring,
            n_folds,
            n_jobs,
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



def unpackRawParams(self, resultsRaw):
    """
    Documentation:
        Definition:
            Unpack results data into a dictionary with model names as the keys and
            Pandas DataFrames as the values. In each DataFrame, there is one column per
            parameter evaluated in the associated model.
        Parameters
            resultsRaw : Pandas DataFrame
                Pandas DataFrame containing results from Bayesian Optimization.
        Returns:
            resultsDict : dictionary of Pandas DataFrames
                Dictionary of 'model name : parameter results' key/value pairs.
    """
    
    clfs = resultsRaw["estimator"].unique()
    resultsDict = {}
    for clf in clfs:
        resultsClf = resultsRaw[resultsRaw["estimator"] == clf].reset_index(drop=True)

        # Create a new dataframe for storing parameters
        paramsClf = pd.DataFrame(
            columns=list(ast.literal_eval(resultsClf.loc[0, "params"]).keys()),
            index=list(range(len(resultsClf))),
        )

        # Add the results with each parameter a different column
        for i, params in enumerate(resultsClf["params"]):
            paramsClf.loc[i, :] = list(ast.literal_eval(params).values())

        # add columns for loss and iter number
        paramsClf["iterLoss"] = resultsClf["loss"]
        paramsClf["iteration"] = resultsClf["iteration"]

        resultsDict[clf] = paramsClf
    return resultsDict


def lossPlot(self, resultsAsDict, chartProp=15, trimOutliers = True, outlierControl=1.5):
    """
    Documentation:
        Definition:
            Visualize how the Bayesian Optimization loss change over time across all iterations.
            Extremely poor results are removed from visualized dataset by two filters.
                1) Loss values worse than [loss mean + (2 x loss std)]
                2) Los values worse than [median * outliersControl]. 'outlierControl' is a parameter
                   that can be set during function execution.
        Parameters
            resultsAsDict : dictionary
                Dictionary of 'model : Pandas DataFrame' pairs, where the DataFrame contains
                the parameter set utilized in each iteration for the model specified as the key.
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
    """
    for clf, df in resultsAsDict.items():
        
        # build dataset to be plotted
        lossDf = pd.DataFrame(
            np.stack((df["iterLoss"], df["iteration"]), axis=-1),
            columns=["iterLoss", "iteration"],
        )

        if trimOutliers:
            mean = lossDf['iterLoss'].mean()
            median = lossDf['iterLoss'].median()        
            std = lossDf['iterLoss'].std()
            cap = mean + (2.0*std)
            lossDf = lossDf[(lossDf['iterLoss'] < cap) & (lossDf['iterLoss'] < outlierControl * median)]

            print(median)
            print(outlierControl)
            print(outlierControl * median)
        
        
        # create regression plot
        p = PrettierPlot(chartProp=chartProp)
        ax = p.makeCanvas(
            title="Regression plot\n* {}".format(clf), yShift=0.8, position=111
        )
        p.prettyRegPlot(x="iteration",
            y="iterLoss",
            data=lossDf,
            yUnits="ffff",
            ax=ax
        )


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
            color=style.styleHexMid[0],
            yUnits="p",
            xUnits="fff" if np.max(theoreticalDist) <= 5.0 else "ff",
            ax=ax,
        )


def paramPlot(self, resultsAsDict, allSpace, nIter, chartProp = 10):
    """
    Documentation:
        Definition:
            Visualize hyper-parameter optimization over the iterations. Compares theoretical
            distribution to the distribution of values that were actually chosen. For parameters
            with a continuous range of values, this function also visualizes how the parameter 
            value changes over time.
        Parameters
            resultsAsDict : dictionary
                Dictionary of 'model : Pandas DataFrame' pairs, where the DataFrame contains
                the parameter set utilized in each iteration for the model specified as the key.
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
    """
    # iterate through model name/df result key/value pairs
    for clf, df in resultsAsDict.items():
        # return space belonging to clf
        clfSpace = allSpace[clf]

        print('*' * 100)
        print('* {}'.format(clf))
        print('*' * 100)

        # iterate through each parameter
        for param in clfSpace.keys():
            # create data to represent theoretical distribution
            theoreticalDist = []
            for _ in range(nIter):
                theoreticalDist.append(sample(clfSpace)[param])

            # clean up
            theoreticalDist = ["None" if v is None else v for v in theoreticalDist]
            theoreticalDist = np.array(theoreticalDist)

            # clean up
            actualDist = df[param].tolist()
            actualDist = ["None" if v is None else v for v in actualDist]
            actualDist = np.array(actualDist)

            actualIterDf = pd.DataFrame(
                np.stack((df["iteration"], df[param]), axis=-1),
                columns=["iteration", param],
            )

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
                )
                p.prettyBarV(
                    x=uniqueVals,
                    counts=uniqueCounts,
                    labelRotate=90 if len(uniqueVals) >= 4 else 0,
                    color=style.styleHexMid[2],
                    yUnits="f",
                    ax=ax,
                )

                # actual plot
                uniqueVals, uniqueCounts = np.unique(actualDist, return_counts=True)
                if param == 'loss':
                    print(uniqueVals)
                    print(uniqueCounts)

                ax = p.makeCanvas(
                    title="Actual plot\n* {}".format(param), yShift=0.8, position=122
                )
                p.prettyBarV(
                    x=uniqueVals,
                    counts=uniqueCounts,
                    labelRotate=90 if len(uniqueVals) >= 4 else 0,
                    color=style.styleHexMid[2],
                    yUnits="f",
                    ax=ax,
                )

                plt.show()

            # plot distributions for continuous params
            else:
                # using dictionary to convert specific columns
                convert_dict = {"iteration": int, param: float}

                actualIterDf = actualIterDf.astype(convert_dict)

                p = PrettierPlot(chartProp=chartProp)
                # p = PrettierPlot(chartProp=8, plotOrientation="square")
                ax = p.makeCanvas(
                    title="Actual vs. theoretical plot\n* {}".format(param),
                    yShift=0.8,
                    position=121,
                )
                p.prettyKdePlot(
                    theoreticalDist,
                    color=style.styleHexMid[0],
                    yUnits="ppp",
                    xUnits="fff" if np.max(theoreticalDist) <= 5.0 else "ff",
                    ax=ax,
                )
                p.prettyKdePlot(
                    actualDist,
                    color=style.styleHexMid[1],
                    yUnits="ppp",
                    xUnits="fff" if np.max(actualDist) <= 5.0 else "ff",
                    ax=ax,
                )

                ## create custom legend
                # create labels
                labelColor = {}
                legendLabels = ['Theoretical','Actual']
                for ix, i in enumerate(legendLabels):
                    labelColor[i] = style.styleHexMid[ix]

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
                    title="Regression plot\n* {}".format(clf), yShift=0.8, position=122
                )
                p.prettyRegPlot(
                    x="iteration", y=param, data=actualIterDf, yUnits="fff", ax=ax
                )
                plt.show()

