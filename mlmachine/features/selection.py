import matplotlib.pyplot as plt

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


class FeatureSync(base.TransformerMixin, base.BaseEstimator):
    """
    Documentation:
        Description:
            Intended to be used on test/validation datasets to ensure that the features in the
            training set are also in the test/validation data sets, and also ensures the features
            are in the same order in all datasets. The issue is handled in two ways. First, ordinal
            feature levels that are in the training data but not in the test/validation datasets
            are added in as all-zero features. Second, there may be features in the test/validation
            datasets but not the training data. This occurs if features were dropped in the training
            data but not yet dropped in the test/validation datasets.
        Parameters:
            trainCols : list
                List containing the columns of the training dataset that have already been
                transformed using pd.get_dummies().
        Returns:
            X : array
                Dataset with dummy column representation of input variables.
    """

    def __init__(self, trainCols):
        self.trainCols = trainCols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # add in missing levels
        missingLevels = set(self.trainCols) - set(X.columns)
        for c in missingLevels:
            X[c] = 0

        # arrange features in same order, and drop any features not present in training data
        X = X[self.trainCols]
        return X

class FeatureSelector():
    """
    Documentation:
        Description:

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
    """
    def __init__(self, data, target, estimators, rank=True, classification=True):
        self.data = data
        self.target = target
        self.estimators = estimators
        self.rank = rank
        self.classification = classification


    def featureSelectorSummary(self):
        """
        Documentation:
            Description:
                Run all feature selections processes and aggregate results. Calculate summary
                statistics on results.
        """
        # run individual top feature processes
        self.resultsVariance = self.featureSelectorVariance()
        self.resultsImportance = self.featureSelectorImportance(estimators=self.estimators, rank=self.rank)
        self.resultsRFE = self.featureSelectorRFE(estimators=self.estimators, rank=self.rank)
        self.resultsCorr = self.featureSelectorCorr()
        if self.classification:
            self.resultsFScore = self.featureSelectorFScoreClass()
        else:
            self.resultsFScore = self.featureSelectorFScoreReg()

        # combine results into single summary table
        results = [self.resultsFScore, self.resultsVariance, self.resultsCorr, self.resultsRFE, self.resultsImportance]
        self.resultsSummary = pd.concat(results, join="inner", axis=1)

        # add summary stats
        self.resultsSummary.insert(loc=0, column="average", value=self.resultsSummary.mean(axis=1))
        self.resultsSummary.insert(loc=1, column="stdev", value=self.resultsSummary.iloc[:, 1:].std(axis=1))
        self.resultsSummary.insert(loc=2, column="low", value=self.resultsSummary.iloc[:, 2:].min(axis=1))
        self.resultsSummary.insert(loc=3, column="high", value=self.resultsSummary.iloc[:, 3:].max(axis=1))

        self.resultsSummary = self.resultsSummary.sort_values("average")


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
            ] = model.feature_importances(data.values, target)

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


    def featureSelectorCorr(self, data=None, target=None, targetName=None, rank=True):
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
            columns={targetName: "TargetCorrelation"}
        )

        featureDf = featureDf.sort_values(
            "TargetCorrelation", ascending=False
        )
        featureDf = featureDf.drop(targetName, axis=0)

        # overwrite values with rank where lower ranks convey higher importance
        if self.rank:
            featureDf = featureDf.rank(ascending=False, method="max")

        return featureDf


    # def featureSelectorCrossVal(self, estimators, featureSummary, metrics, data=None, target=None, nFolds=3, step=1,
    #                             nJobs=4, verbose=True):
    #     """
    #     Documentation:
    #         Description:
    #             Perform cross-validation for each estimator, for progressively smaller sets of features. The list
    #             of features is reduced by one feature on each pass. The feature removed is the least important
    #             feature of the remaining set. Calculates both the training and test performance.
    #         Parameters:
    #             estimators : list of strings or sklearn API objects.
    #                 List of estimators to be used.
    #             featureSummary : Pandas DataFrame
    #                 Feature importance summary table.
    #             mtrics : list of strings
    #                 List containing strings for one or more performance metrics.
    #             data : Pandas DataFrame, default = None
    #                 Pandas DataFrame containing independent variables. If left as None,
    #                 the feature dataset provided to Machine during instantiation is used.
    #             target : Pandas Series, default = None
    #                 Pandas Series containing dependent target variable. If left as None,
    #                 the target dataset provided to Machine during instantiation is used.
    #             nFolds : int, default = 3
    #                 Number of folds to use in cross validation.
    #             step : int, default = 1
    #                 Number of features to remove per iteration.
    #             nJobs : int, default = 4
    #                 Number of works to use when training the model. This parameter will be
    #                 ignored if the model does not have this parameter.
    #             verbose : boolean, default = True
    #                 Conditional controlling whether each estimator name is printed prior to cross-validation.
    #         Return:
    #             cvSummary : dictionary
    #                 Dictionary containing string/Pandas DataFrame key/value pairs, where the
    #                 key is an estimator and the value is a Pandas DataFrame summarizing the
    #                 training and validation performance for each feature set.
    #     """
    #     # create empty dictionary for capturing one DataFrame for each estimator
    #     cvSummary = {}

    #     # perform cross validation for all estimators for each diminishing set of features
    #     for estimator in self.estimators:

    #         if verbose:
    #             print(estimator)

    #         # instantiate default model and create empty DataFrame for capturing scores
    #         model = BasicModelBuilder(estimator=estimator, nJobs=nJobs)
    #         cv = pd.DataFrame(columns=["Training score", "Validation score","scoring"])
    #         rowIx = 0

    #         # iterate through scoring metrics
    #         for metric in metrics:
    #             # iterate through each set of features
    #             for i in np.arange(0, featureSummary.shape[0], step):
    #                 if i ==0:
    #                     top = featureSummary.sort_values("average").index
    #                 else:
    #                     top = featureSummary.sort_values("average").index[:-i]
    #                 scores = model_selection.cross_validate(
    #                     estimator=model.model,
    #                     X=self.data[top],
    #                     y=target,
    #                     cv=nFolds,
    #                     scoring=metric,
    #                     return_train_score=True,
    #                 )

    #                 # calculate mean scores
    #                 training = scores["train_score"].mean()
    #                 validation = scores["test_score"].mean()

    #                 # append results
    #                 cv.loc[rowIx] = [training, validation, metric]
    #                 rowIx += 1

    #         # capturing results DataFrame associated with estimator
    #         cvSummary[estimator] = cv
    #     return cvSummary


    # def featureSelectorResultsPlot(self, cvSummary, featureSummary, metric, topSets=0, showFeatures=False, showScores=False, markerOn=True,
    #                                 titleScale=0.7):
    #     """
    #     Documentation:
    #         Description:
    #             For each estimator, visualize the training and validation performance
    #             for each feature set.
    #         Parameters:
    #             cvSummary : dictionary
    #                 Dictionary containing string/Pandas DataFrame key/value pairs, where the
    #                 key is an estimator and the value is a Pandas DataFrame summarizing the
    #                 training and validation performance for each feature set.
    #             featureSummary : Pandas DataFrame
    #                 Feature importance summary table.
    #             metric : string
    #                 Metric to visualize.
    #             topSets : int, default = 5
    #                 Number of rows to display of the performance summary table
    #             showFeatures : boolean, default = False
    #                 Conditional controlling whether to print feature set for best validation
    #                 score.
    #             showScores : boolean, default = False
    #                 Conditional controlling whether to display training and validation scores.
    #             markerOn : boolean, default = True
    #                 Conditional controlling whether to display marker for each individual score.
    #             titleScale : float, default = 1.0
    #                 Controls the scaling up (higher value) and scaling down (lower value) of the size of
    #                 the main chart title, the x-axis title and the y-axis title.
    #     """
    #     for estimator in cvSummary.keys():
    #         cv = cvSummary[estimator][cvSummary[estimator]['scoring'] == metric]

    #         totalFeatures = featureSummary.shape[0]
    #         iters = cv.shape[0]
    #         step = np.ceil(totalFeatures / iters)

    #         cv.set_index(keys=np.arange(0, cv.shape[0] * step, step, dtype=int), inplace=True)

    #         if showScores:
    #             display(cv[:5])

    #         # capture best iteration's feature drop count and performance score
    #         numDropped = (
    #             cv
    #             .sort_values(["Validation score"], ascending=False)[:1]
    #             .index.values[0]
    #         )
    #         score = np.round(
    #             cv
    #             .sort_values(["Validation score"], ascending=False)["Validation score"][:1]
    #             .values[0],
    #             5,
    #         )

    #         # display performance for the top N feature sets
    #         if topSets > 0:
    #             display(cv.sort_values(["Validation score"], ascending=False)[:topSets])
    #         if showFeatures:
    #             if numDropped > 0:
    #                 featuresUsed = featureSummary.sort_values("average").index[:-numDropped].values
    #             else:
    #                 featuresUsed = featureSummary.sort_values("average").index.values
    #             print(featuresUsed)

    #         # create multi-line plot
    #         p = PrettierPlot()
    #         ax = p.makeCanvas(
    #             title="{}\nBest validation {} = {}\nFeatures dropped = {}".format(
    #                 estimator, metric, score, numDropped
    #             ),
    #             xLabel="Features removed",
    #             yLabel=metric,
    #             yShift=0.4 if len(metric) > 18 else 0.57,
    #             titleScale=titleScale
    #         )

    #         p.prettyMultiLine(
    #             x=cv.index,
    #             y=["Training score", "Validation score"],
    #             label=["Training score", "Validation score"],
    #             df=cv,
    #             yUnits="fff",
    #             markerOn=markerOn,
    #             bbox=(1.3, 0.9),
    #             ax=ax,
    #         )
    #         plt.show()


    # def featuresUsedSummary(self, cvSummary, metric, featureSummary):
    #     """
    #     Documentation:
    #         Description:
    #             For each estimator, visualize the training and validation performance
    #             for each feature set.
    #         Parameters:
    #             cvSummary : dictionary
    #                 Dictionary containing string/Pandas DataFrame key/value pairs, where the
    #                 key is an estimator and the value is a Pandas DataFrame summarizing the
    #                 training and validation performance for each feature set.
    #             metric : string
    #                 Metric to visualize.
    #             featureSummary : Pandas DataFrame
    #                 Feature importance summary table.

    #     """
    #     # create empty DataFrame with feature names as index
    #     df = pd.DataFrame(index=featureSummary.index)

    #     # iterate through estimators
    #     for estimator in cvSummary.keys():
    #         cv = cvSummary[estimator][cvSummary[estimator]['scoring'] == metric]
    #         cv = cv.reset_index(drop=True)

    #         # capture best iteration's feature drop count
    #         numDropped = (
    #             cv
    #             .sort_values(["Validation score"], ascending=False)[:1]
    #             .index.values[0]
    #         )
    #         if numDropped > 0:
    #             featuresUsed = featureSummary.sort_values("average").index[:-numDropped].values
    #         else:
    #             featuresUsed = featureSummary.sort_values("average").index.values

    #         # create column for estimator and populate with marker
    #         df[estimator.split(".")[1]] = np.nan
    #         df[estimator.split(".")[1]].loc[featuresUsed] = "X"

    #     # add counter and fill NaNs
    #     df["count"] = df.count(axis=1)
    #     df = df.fillna("")
    #     return df