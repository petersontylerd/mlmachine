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


from ..model.tune.bayesianOptimSearch import (
        BasicModelBuilder
    )


class FeatureSelector():
    """
    Documentation:
        Description:
            Evaluate feature importance using several different feature selection techniques,
            including F-score, variance, recursive feature selection, and correlation to target
            on a list of estimators. Also includes methods for performing corss-validation and
            visualization of the results.P
        Parameters:
            data : Pandas DataFrame, default = None
                Pandas DataFrame containing independent variables. If left as None,
                the feature dataset provided to Machine during instantiation is used.
            target : Pandas Series, default = None
                Pandas Series containing dependent target variable. If left as None,
                the target dataset provided to Machine during instantiation is used.
            estimators : list of strings or sklearn API objects.
                    List of estimators to be used.
            rank : boolean, default = True
                Conditional controlling whether to overwrite values with rank of values.
            classification : boolean, default = True
                Conditional controlling whether object is informed that the supervised learning
                task is a classification task.
    """
    def __init__(self, data, target, estimators, rank=True, classification=True):
        self.data = data
        self.target = target
        self.estimators = estimators
        self.rank = rank
        self.classification = classification


    def featureSelectorSuite(self, save_to_csv=True):
        """
        Documentation:
            Description:
                Run all feature selections processes and aggregate results. Calculate summary
                statistics on results.
            Parameters:
                save_to_csv : boolean, default = True
                    Conditional controlling whethor or not the feature selection summary results
                    are saved to a csv file.
        """
        # run individual top feature processes
        self.resultsVariance = self.featureSelectorVariance()
        self.resultsImportance = self.featureSelectorImportance()
        self.resultsRFE = self.featureSelectorRFE()
        self.resultsCorr = self.featureSelectorCorr()
        if self.classification:
            self.resultsFScore = self.featureSelectorFScoreClass()
        else:
            self.resultsFScore = self.featureSelectorFScoreReg()

        # combine results into single summary table
        results = [self.resultsFScore, self.resultsVariance, self.resultsCorr, self.resultsRFE, self.resultsImportance]
        self.featureSelectorSummary = pd.concat(results, join="inner", axis=1)

        # add summary stats
        self.featureSelectorSummary.insert(loc=0, column="average", value=self.featureSelectorSummary.mean(axis=1))
        self.featureSelectorSummary.insert(loc=1, column="stdev", value=self.featureSelectorSummary.iloc[:, 1:].std(axis=1))
        self.featureSelectorSummary.insert(loc=2, column="low", value=self.featureSelectorSummary.iloc[:, 2:].min(axis=1))
        self.featureSelectorSummary.insert(loc=3, column="high", value=self.featureSelectorSummary.iloc[:, 3:].max(axis=1))

        self.featureSelectorSummary = self.featureSelectorSummary.sort_values("average")

        if save_to_csv:
            self.featureSelectorSummary.to_csv(
                "featureSelectionSummary_{}.csv".format(strftime("%Y%m%d_%H%M%S", gmtime())),
                columns=self.featureSelectorSummary.columns,
                # index_label="index"
            )
        return self.featureSelectorSummary


    def featureSelectorFScoreClass(self):
        """
        Documentation:
            Description:
                For each feature, calculate F-values and p-values in the context of a
                classification probelm.
        """
        # calculate F-values and p-values
        univariate = feature_selection.f_classif(self.data, self.target)

        # Parse data into dictionary
        featureDict = {}
        featureDict["F-value"] = univariate[0]
        featureDict["p-value"] = univariate[1]

        # load dictionary into Pandas DataFrame and rank values
        featureDf = pd.DataFrame(data=featureDict, index=self.data.columns)

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            featureDf["F-value"] = featureDf["F-value"].rank(ascending=False, method="max")
            featureDf["p-value"] = featureDf["p-value"].rank(ascending=True, method="max")

        return featureDf


    def featureSelectorFScoreReg(self):
        """
        Documentation:
            Description:
                For each feature, calculate F-values and p-values in the context of a
                regression probelm.
        """
        # calculate F-values and p-values
        univariate = feature_selection.f_regression(self.data, self.target)

        # Parse data into dictionary
        featureDict = {}
        featureDict["F-value"] = univariate[0]
        featureDict["p-value"] = univariate[1]

        # load dictionary into Pandas DataFrame and rank values
        featureDf = pd.DataFrame(data=featureDict, index=self.data.columns)

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            featureDf["F-value"] = featureDf["F-value"].rank(ascending=False, method="max")
            featureDf["p-value"] = featureDf["p-value"].rank(ascending=True, method="max")

        return featureDf


    def featureSelectorVariance(self):
        """
        Documentation:
            Description:
                For each feature, calculate variance.
        """
        # calculate variance
        varImportance = feature_selection.VarianceThreshold()
        varImportance.fit(self.data)

        # load data into Pandas DataFrame and rank values
        featureDf = pd.DataFrame(
            varImportance.variances_, index=self.data.columns, columns=["Variance"]
        )

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            featureDf['Variance'] = featureDf['Variance'].rank(ascending=False, method="max")

        return featureDf


    def featureSelectorImportance(self):
        """
        Documentation:
            Description:
                For each estimator, for each feature, calculate feature importance.
        """
        #
        featureDict = {}
        for estimator in self.estimators:
            model = BasicModelBuilder(estimator=estimator)
            featureDict[
                "FeatureImportance " + model.estimator.__name__
            ] = model.feature_importances(self.data.values, self.target)

        featureDf = pd.DataFrame(featureDict, index=self.data.columns)

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            featureDf = featureDf.rank(ascending=False, method="max")

        return featureDf


    def featureSelectorRFE(self):
        """
        Documentation:
            Description:
                For each estimator, recursively remove features one at a time, capturing
                the step in which each feature is removed.
        """
        #
        featureDict = {}
        for estimator in self.estimators:
            model = BasicModelBuilder(estimator=estimator)

            # recursive feature selection
            rfe = feature_selection.RFE(
                estimator=model.model, n_features_to_select=1, step=1, verbose=0
            )
            rfe.fit(self.data, self.target)
            featureDict["RFE " + model.estimator.__name__] = rfe.ranking_

        featureDf = pd.DataFrame(featureDict, index=self.data.columns)

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            featureDf = featureDf.rank(ascending=True, method="max")

        return featureDf


    def featureSelectorCorr(self):
        """
        Documentation:
            Description:
                For each feature, calculate absolute correlation coefficient relative to
                target dataset.
        """
        # calculate absolute correlation coefficients relative to target
        featureDf = self.data.merge(self.target, left_index=True, right_index=True)

        featureDf = pd.DataFrame(
            featureDf.corr().abs()[self.target.name]
        )
        featureDf = featureDf.rename(
            columns={self.target.name: "TargetCorrelation"}
        )

        featureDf = featureDf.sort_values(
            "TargetCorrelation", ascending=False
        )
        featureDf = featureDf.drop(self.target.name, axis=0)

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            featureDf = featureDf.rank(ascending=False, method="max")

        return featureDf


    def featureSelectorCrossVal(self, scoring, featureSelectorSummary=None, estimators=None, nFolds=3, step=1, nJobs=4, verbose=True, save_to_csv=True):
        """
        Documentation:
            Description:
                Perform cross-validation for each estimator, for progressively smaller sets of features. The list
                of features is reduced by one feature on each pass. The feature removed is the least important
                feature of the remaining set. Calculates both the training and test performance.
            Parameters:
                scoring : list of strings
                    List containing strings for one or more performance scoring metrics.
                featureSelectorSummary : Pandas DataFrame or str, default = None
                    Pandas DataFrame, or str of csv file location, containing summary of featureSelectorSuite results.
                    If none, use object's internal attribute specified during instantiation.
                estimators : list of strings or sklearn API objects, default = None
                    List of estimators to be used. If none, use object's internal attribute specified during instantiation.
                nFolds : int, default = 3
                    Number of folds to use in cross validation.
                step : int, default = 1
                    Number of features to remove per iteration.
                nJobs : int, default = 4
                    Number of works to use when training the model. This parameter will be
                    ignored if the model does not have this parameter.
                verbose : boolean, default = True
                    Conditional controlling whether each estimator name is printed prior to cross-validation.
                save_to_csv : boolean, default = True
                    Conditional controlling whethor or not the feature selection summary results
                    are saved to a csv file.
        """
        # load results summary if needed
        if isinstance(featureSelectorSummary, str):
            featureSelectorSummary=pd.read_csv(featureSelectorSummary, index_col=0)
        elif isinstance(featureSelectorSummary, pd.core.frame.DataFrame):
            featureSelectorSummary=featureSelectorSummary
        elif featureSelectorSummary is None:
            try:
                featureSelectorSummary=self.featureSelectorSummary
            except AttributeError:
                raise AttributeError("No featureSelectorSummary detected. Either execute method featureSelectorSuite or load from CSV")

        # load estimators if needed
        if estimators is None:
            estimators=self.estimators

        # create empty dictionary for capturing one DataFrame for each estimator
        self.cvSummary = pd.DataFrame(columns=["Estimator","Training score","Validation score","Scoring","Features dropped"])

        # perform cross validation for all estimators for each diminishing set of features
        rowIx = 0

        for estimator in estimators:

            if verbose:
                print(estimator)

            # instantiate default model and create empty DataFrame for capturing scores
            model = BasicModelBuilder(estimator=estimator, nJobs=nJobs)
            cv = pd.DataFrame(columns=["Estimator","Training score","Validation score","Scoring","Features dropped"])

            # iterate through scoring metrics
            for metric in scoring:

                stepIncrement = 0
                # iterate through each set of features
                for i in np.arange(0, featureSelectorSummary.shape[0], step):
                    # collect names of top columns
                    if i ==0:
                        top = featureSelectorSummary.sort_values("average").index
                    else:
                        top = featureSelectorSummary.sort_values("average").index[:-i]

                    # custom metric handling
                    if metric == "root_mean_squared_error":
                        metric = "neg_mean_squared_error"
                        scoreTransform = "rmse"
                    elif metric == "root_mean_squared_log_error":
                        metric = "neg_mean_squared_log_error"
                        scoreTransform = "rmsle"
                    else:
                        scoreTransform = metric


                    scores = model_selection.cross_validate(
                        estimator=model.model,
                        X=self.data[top],
                        y=self.target,
                        cv=nFolds,
                        scoring=metric,
                        return_train_score=True,
                    )

                    # custom metric handling
                    if scoreTransform == "rmse":
                        training = np.mean(np.sqrt(np.abs(scores["train_score"])))
                        validation = np.mean(np.sqrt(np.abs(scores["test_score"])))
                        metric = "root_mean_squared_error"
                    elif scoreTransform == "rmsle":
                        training = np.mean(np.sqrt(np.abs(scores["train_score"])))
                        validation = np.mean(np.sqrt(np.abs(scores["test_score"])))
                        metric = "root_mean_squared_log_error"
                    else:
                        training = np.mean(scores["train_score"])
                        validation = np.mean(scores["test_score"])

                    # append results
                    cv.loc[rowIx] = [model.estimator.__name__, training, validation, metric, stepIncrement]
                    stepIncrement += step
                    rowIx += 1

            # capturing results DataFrame associated with estimator
            self.cvSummary = self.cvSummary.append(cv)
            # self.cvSummary[estimator] = cv

        if save_to_csv:
            self.cvSummary.to_csv("cvSummary_{}.csv".format(strftime("%Y%m%d_%H%M%S", gmtime())), columns=self.cvSummary.columns, index_label="index")

        return self.cvSummary


    def featureSelectorResultsPlot(self, metric, cvSummary=None, featureSelectorSummary=None, topSets=0, showFeatures=False, showScores=None, markerOn=True, titleScale=0.7):
        """
        Documentation:
            Description:
                For each estimator, visualize the training and validation performance
                for each feature set.
            Parameters:
                metric : string
                    Metric to visualize.
                cvSummary : Pandas DataFrame or str, default = None
                    Pandas DataFrame, or str of csv file location, containing cross-validation results.
                    If none, use object's internal attribute specified during instantiation.
                featureSelectorSummary : Pandas DataFrame or str, default = None
                    Pandas DataFrame, or str of csv file location, containing summary of featureSelectorSuite results.
                    If none, use object's internal attribute specified during instantiation.
                topSets : int, default = 5
                    Number of rows to display of the performance summary table
                showFeatures : boolean, default = False
                    Conditional controlling whether to print feature set for best validation
                    score.
                showScores : int or None, default = None
                    Display certain number of top features. If None, display nothing. If int, display
                    the specified number of features as a Pandas DataFrame.
                markerOn : boolean, default = True
                    Conditional controlling whether to display marker for each individual score.
                titleScale : float, default = 1.0
                    Controls the scaling up (higher value) and scaling down (lower value) of the size of
                    the main chart title, the x-axis title and the y-axis title.
        """
        # load results summary if needed
        if isinstance(featureSelectorSummary, str):
            featureSelectorSummary=pd.read_csv(featureSelectorSummary, index_col=0)
        elif isinstance(featureSelectorSummary, pd.core.frame.DataFrame):
            featureSelectorSummary=featureSelectorSummary
        elif featureSelectorSummary is None:
            try:
                featureSelectorSummary=self.featureSelectorSummary
            except AttributeError:
                raise AttributeError("No featureSelectorSummary detected. Either execute method featureSelectorSuite or load from CSV")

        # load CV summary if needed
        if isinstance(cvSummary, str):
            cvSummary=pd.read_csv(cvSummary, index_col=0)
        elif isinstance(cvSummary, pd.core.frame.DataFrame):
            cvSummary=cvSummary
        elif cvSummary is None:
            try:
                cvSummary=self.cvSummary
            except AttributeError:
                raise AttributeError("no cvSummary detected. Either execute method featureSelectorCrossVal or load from CSV")

        for estimator in cvSummary["Estimator"].unique():
            cv = cvSummary[(cvSummary['Scoring'] == metric) & (cvSummary['Estimator'] == estimator)]

            totalFeatures = featureSelectorSummary.shape[0]
            iters = cv.shape[0]
            step = np.ceil(totalFeatures / iters)

            cv.set_index(keys=np.arange(0, cv.shape[0] * step, step, dtype=int), inplace=True)

            if showScores is not None:
                display(cv[:showScores])

            # capture best iteration's feature drop count and performance score
            if metric in ["root_mean_squared_error","root_mean_squared_log_error"]:
                sortOrder = True
            else:
                sortOrder = False

            numDropped = (
                cv
                .sort_values(["Validation score"], ascending=sortOrder)[:1]["Features dropped"]
                .values[0]
            )
            score = np.round(
                cv
                .sort_values(["Validation score"], ascending=sortOrder)["Validation score"][:1]
                .values[0],
                5,
            )

            # display performance for the top N feature sets
            if topSets > 0:
                display(cv.sort_values(["Validation score"], ascending=sortOrder)[:topSets])
            if showFeatures:
                if numDropped > 0:
                    features = featureSelectorSummary.shape[0] - numDropped
                    featuresUsed = featureSelectorSummary.sort_values("average").index[:features].values
                else:
                    featuresUsed = featureSelectorSummary.sort_values("average").index.values
                print(featuresUsed)

            # create multi-line plot
            p = PrettierPlot()
            ax = p.makeCanvas(
                title="{}\nBest validation {} = {}\nFeatures dropped = {}".format(
                    estimator, metric, score, numDropped
                ),
                xLabel="Features removed",
                yLabel=metric,
                yShift=0.4 if len(metric) > 18 else 0.57,
                titleScale=titleScale
            )

            p.prettyMultiLine(
                x=cv.index,
                y=["Training score", "Validation score"],
                label=["Training score", "Validation score"],
                df=cv,
                yUnits="fff",
                markerOn=markerOn,
                bbox=(1.3, 0.9),
                ax=ax,
            )
            plt.show()


    def featuresUsedSummary(self, metric, cvSummary=None, featureSelectorSummary=None):
        """
        Documentation:
            Description:
                For each estimator, visualize the training and validation performance
                for each feature set.
            Parameters:
                metric : string
                    Metric to visualize.
                cvSummary : Pandas DataFrame or str, default = None
                    Pandas DataFrame, or str of csv file location, containing cross-validation results.
                    If none, use object's internal attribute specified during instantiation.
                featureSelectorSummary : Pandas DataFrame or str, default = None
                    Pandas DataFrame, or str of csv file location, containing summary of featureSelectorSuite results.
                    If none, use object's internal attribute specified during instantiation.
        """
        # load results summary if needed
        if isinstance(featureSelectorSummary, str):
            featureSelectorSummary=pd.read_csv(featureSelectorSummary, index_col=0)
        elif isinstance(featureSelectorSummary, pd.core.frame.DataFrame):
            featureSelectorSummary=featureSelectorSummary
        elif featureSelectorSummary is None:
            try:
                featureSelectorSummary=self.featureSelectorSummary
            except AttributeError:
                raise AttributeError("No featureSelectorSummary detected. Either execute method featureSelectorSuite or load from CSV")

        # load CV summary if needed
        if isinstance(cvSummary, str):
            cvSummary=pd.read_csv(cvSummary, index_col=0)
        elif isinstance(cvSummary, pd.core.frame.DataFrame):
            cvSummary=cvSummary
        elif cvSummary is None:
            try:
                cvSummary=self.cvSummary
            except AttributeError:
                raise AttributeError("No cvSummary detected. Either execute method featureSelectorSuite or load from CSV")

        # create empty DataFrame with feature names as index
        df = pd.DataFrame(index=featureSelectorSummary.index)

        # iterate through estimators
        for estimator in cvSummary["Estimator"].unique():
            cv = cvSummary[(cvSummary['Scoring'] == metric) & (cvSummary['Estimator'] == estimator)]
            cv = cv.reset_index(drop=True)

            # capture best iteration's feature drop count
            if metric in ["root_mean_squared_error","root_mean_squared_log_error"]:
                sortOrder = True
            else:
                sortOrder = False

            numDropped = (
                cv
                .sort_values(["Validation score"], ascending=sortOrder)[:1]["Features dropped"]
                .values[0]
            )
            if numDropped > 0:
                features = featureSelectorSummary.shape[0] - numDropped
                featuresUsed = featureSelectorSummary.sort_values("average").index[:features].values
            else:
                featuresUsed = featureSelectorSummary.sort_values("average").index.values

            # create column for estimator and populate with marker
            df[estimator] = np.nan
            df[estimator].loc[featuresUsed] = "X"

        # add counter and fill NaNs
        df["count"] = df.count(axis=1)
        df = df.fillna("")

        # add numerical index starting at 1
        df = df.reset_index()
        df.index = np.arange(1, len(df) + 1)
        df = df.rename(columns={'index': 'Feature'})
        return df