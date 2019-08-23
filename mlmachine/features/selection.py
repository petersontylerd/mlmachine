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



def featureSelectorFScoreClass(self, data=None, target=None, rank=True):
    """
    Documentation:
        Description:
            For each feature, calculate F-values and p-values in the context of a
            classification probelm.
        Parameters:
            data : Pandas DataFrame, default = None
                Pandas DataFrame containing independent variables. If left as None,
                the feature dataset provided to Machine during instantiation is used.
            target : Pandas Series, default = None
                Pandas Series containing dependent target variable. If left as None,
                the target dataset provided to Machine during instantiation is used.
            rank : boolean, default = True
                Conditional controlling whether to overwrite values with rank of values.
        Return:
            featureDf : Pandas DataFrame
                Pandas DataFrame containing summary of F-values and p-values by feature.
    """
    # use data/target provided during instantiation if left unspecified
    if data is None:
        data = self.data
    if target is None:
        target = self.target

    # calculate F-values and p-values
    univariate = feature_selection.f_classif(data, target)

    # Parse data into dictionary
    featureDict = {}
    featureDict["F-value"] = univariate[0]
    featureDict["p-value"] = univariate[1]

    # load dictionary into Pandas DataFrame and rank values
    featureDf = pd.DataFrame(data=featureDict, index=data.columns)

    # overwrite values with rank where lower ranks convey higher importance
    if rank:
        featureDf["F-value"] = featureDf["F-value"].rank(ascending=False, method="max")
        featureDf["p-value"] = featureDf["p-value"].rank(ascending=True, method="max")

    return featureDf

def featureSelectorFScoreReg(self, data=None, target=None, rank=True):
    """
    Documentation:
        Description:
            For each feature, calculate F-values and p-values in the context of a
            regression probelm.
        Parameters:
            data : Pandas DataFrame, default = None
                Pandas DataFrame containing independent variables. If left as None,
                the feature dataset provided to Machine during instantiation is used.
            target : Pandas Series, default = None
                Pandas Series containing dependent target variable. If left as None,
                the target dataset provided to Machine during instantiation is used.
            rank : boolean, default = True
                Conditional controlling whether to overwrite values with rank of values.
        Return:
            featureDf : Pandas DataFrame
                Pandas DataFrame containing summary of F-values and p-values by feature.
    """
    # use data/target provided during instantiation if left unspecified
    if data is None:
        data = self.data
    if target is None:
        target = self.target

    # calculate F-values and p-values
    univariate = feature_selection.f_regression(data, target)

    # Parse data into dictionary
    featureDict = {}
    featureDict["F-value"] = univariate[0]
    featureDict["p-value"] = univariate[1]

    # load dictionary into Pandas DataFrame and rank values
    featureDf = pd.DataFrame(data=featureDict, index=data.columns)

    # overwrite values with rank where lower ranks convey higher importance
    if rank:
        featureDf["F-value"] = featureDf["F-value"].rank(ascending=False, method="max")
        featureDf["p-value"] = featureDf["p-value"].rank(ascending=True, method="max")

    return featureDf

def featureSelectorVariance(self, data=None, target=None, rank=True):
    """
    Documentation:
        Description:
            For each feature, calculate variance.
        Parameters:
            data : Pandas DataFrame, default = None
                Pandas DataFrame containing independent variables. If left as None,
                the feature dataset provided to Machine during instantiation is used.
            target : Pandas Series, default = None
                Pandas Series containing dependent target variable. If left as None,
                the target dataset provided to Machine during instantiation is used.
            rank : boolean, default = True
                Conditional controlling whether to overwrite values with rank of values.
        Return:
            featureDf : Pandas DataFrame
                Pandas DataFrame containing summary of variances.
    """
    # use data/target provided during instantiation if left unspecified
    if data is None:
        data = self.data
    if target is None:
        target = self.target

    # calculate variance
    varImportance = feature_selection.VarianceThreshold()
    varImportance.fit(self.data)

    # load data into Pandas DataFrame and rank values
    featureDf = pd.DataFrame(
        varImportance.variances_, index=data.columns, columns=["Variance"]
    )

    # overwrite values with rank where lower ranks convey higher importance
    if rank:
        featureDf['Variance'] = featureDf['Variance'].rank(ascending=False, method="max")

    return featureDf

def featureSelectorImportance(self, estimators, data=None, target=None, rank=True):
    """
    Documentation:
        Description:
            For each estimator, for each feature, calculate feature importance.
        Parameters:
            estimators : list of strings or sklearn API objects.
                List of estimators to be used.
            data : Pandas DataFrame, default = None
                Pandas DataFrame containing independent variables. If left as None,
                the feature dataset provided to Machine during instantiation is used.
            target : Pandas Series, default = None
                Pandas Series containing dependent target variable. If left as None,
                the target dataset provided to Machine during instantiation is used.
            rank : boolean, default = True
                Conditional controlling whether to overwrite values with rank of values.
        Return:
            featureDf : Pandas DataFrame
                Pandas DataFrame containing summary of feature importance by estimator
                and by feautre.
    """
    # use data/target provided during instantiation if left unspecified
    if data is None:
        data = self.data
    if target is None:
        target = self.target

    featureDict = {}
    for estimator in estimators:
        model = self.BasicModelBuilder(estimator=estimator)
        featureDict[
            "FeatureImportance " + estimator.split(".")[1]
        ] = model.feature_importances(data, target)

    featureDf = pd.DataFrame(featureDict, index=data.columns)

    # overwrite values with rank where lower ranks convey higher importance
    if rank:
        featureDf = featureDf.rank(ascending=False, method="max")

    return featureDf

def featureSelectorRFE(self, estimators, data=None, target=None, rank=True):
    """
    Documentation:
        Description:
            For each estimator, recursively remove features one at a time, capturing
            the step in which each feature is removed.
        Parameters:
            estimators : list of strings or sklearn API objects.
                List of estimators to be used.
            data : Pandas DataFrame, default = None
                Pandas DataFrame containing independent variables. If left as None,
                the feature dataset provided to Machine during instantiation is used.
            target : Pandas Series, default = None
                Pandas Series containing dependent target variable. If left as None,
                the target dataset provided to Machine during instantiation is used.
            rank : boolean, default = True
                Conditional controlling whether to overwrite values with rank of values.
        Return:
            featureDf : Pandas DataFrame
                Pandas DataFrame containing summary of recursive feature selection by
                estimaor.
    """
    # use data/target provided during instantiation if left unspecified
    if data is None:
        data = self.data
    if target is None:
        target = self.target

    featureDict = {}
    for estimator in estimators:
        model = self.BasicModelBuilder(estimator=estimator)

        # recursive feature selection
        rfe = feature_selection.RFE(
            estimator=model.model, n_features_to_select=1, step=1, verbose=0
        )
        rfe.fit(data, target)
        featureDict["RFE " + estimator.split(".")[1]] = rfe.ranking_

    featureDf = pd.DataFrame(featureDict, index=data.columns)

    # overwrite values with rank where lower ranks convey higher importance
    if rank:
        featureDf = featureDf.rank(ascending=True, method="max")

    return featureDf

def featureSelectorCorr(self, data=None, target=None, targetName=None, rank=True):
    """
    Documentation:
        Description:
            For each feature, calculate absolute correlation coefficient relative to
            target dataset.
        Parameters:
            data : Pandas DataFrame, default = None
                Pandas DataFrame containing independent variables. If left as None,
                the feature dataset provided to Machine during instantiation is used.
            target : Pandas Series, default = None
                Pandas Series containing dependent target variable. If left as None,
                the target dataset provided to Machine during instantiation is used.
            targetName : string, default = None
                String containing name of target variable. If left as None, the target
                name created by Machine during instantiation is used.
            rank : boolean, default = True
                Conditional controlling whether to overwrite values with rank of values.
        Return:
            featureDf : Pandas DataFrame
                Pandas DataFrame containing absolute correlation coefficients by feature.
    """
    # use data/target/targetName provided during instantiation if left unspecified
    if data is None:
        data = self.data
    if target is None:
        target = self.target
    if targetName is None:
        targetName = self.target.name

    # calculate absolute correlation coefficients relative to target
    featureDf = pd.DataFrame(
        self.edaData(data, target).corr().abs()[targetName]
    )
    featureDf = featureDf.rename(
        columns={targetName: "TargetCorrelation"}
    )

    featureDf = featureDf.sort_values(
        "TargetCorrelation", ascending=False
    )
    featureDf = featureDf.drop(targetName, axis=0)

    # overwrite values with rank where lower ranks convey higher importance
    if rank:
        featureDf = featureDf.rank(ascending=False, method="max")

    return featureDf

def featureSelectorSummary(self, estimators, classification=True):
    """
    Documentation:
        Description:
            For each feature, calculate absolute correlation coefficient relative to
            target dataset.
        Parameters:
            estimators : list of strings or sklearn API objects.
                List of estimators to be used.
            classification : boolean, default = True
                Conditional controlling whether the function is calculate F-values and
                p-values in the context of a classifiation task or a regression task.
        Return:
            featureDf : Pandas DataFrame
                Pandas DataFrame containing absolute correlation coefficients by feature.
    """
    # run individual top feature processes
    resultsVariance = self.featureSelectorVariance()
    resultsImportance = self.featureSelectorImportance(estimators=estimators, rank=True)
    resultsRFE = self.featureSelectorRFE(estimators=estimators, rank=True)
    resultsCorr = self.featureSelectorCorr()
    if classification:
        resultsFScore = self.featureSelectorFScoreClass()
    else:
        resultsFScore = self.featureSelectorFScoreReg()

    # combine results into single summary table
    results = [resultsFScore, resultsVariance, resultsCorr, resultsRFE, resultsImportance]
    resultsSummary = pd.concat(results, join="inner", axis=1)

    # add summary stats
    resultsSummary.insert(loc=0, column="average", value=resultsSummary.mean(axis=1))
    resultsSummary.insert(loc=1, column="stdev", value=resultsSummary.iloc[:, 1:].std(axis=1))
    resultsSummary.insert(loc=2, column="low", value=resultsSummary.iloc[:, 2:].min(axis=1))
    resultsSummary.insert(loc=3, column="high", value=resultsSummary.iloc[:, 3:].max(axis=1))

    resultsSummary = resultsSummary.sort_values("average")
    return resultsSummary


def featureSelectorCrossVal(self, estimators, featureSummary, scoring, data=None, target=None, nFolds=3, verbose=True):
    """
    Documentation:
        Description:
            Perform cross-validation for each estimator, for progressively smaller sets of features. The list
            of features is reduced by one feature on each pass. The feature removed is the least important
            feature of the remaining set. Calculates both the training and test performance.
        Parameters:
            estimators : list of strings or sklearn API objects.
                List of estimators to be used.
            featureSummary : Pandas DataFrame
                Feature importance summary table.
            scoring : tuple of strings
                Tuples containing strings for one or more performance metircs.
            data : Pandas DataFrame, default = None
                Pandas DataFrame containing independent variables. If left as None,
                the feature dataset provided to Machine during instantiation is used.
            target : Pandas Series, default = None
                Pandas Series containing dependent target variable. If left as None,
                the target dataset provided to Machine during instantiation is used.
            nFolds : int, default = 3
                Number of folds to use in cross validation.
            verbose : boolean, default = True
                Conditional controlling whether each estimator name is printed prior to cross-validation.
        Return:
            cvSummary : dictionary
                Dictionary containing string/Pandas DataFrame key/value pairs, where the
                key is an estimator and the value is a Pandas DataFrame summarizing the
                training and validation performance for each feature set.
    """
    # use data/target provided during instantiation if left unspecified
    if data is None:
        data = self.data
    if target is None:
        target = self.target

    # create empty dictionary for capturing one DataFrame for each estimator
    cvSummary = {}

    # perform cross validation for all estimators for each diminishing set of features
    for estimator in estimators:

        if verbose:
            print(estimator)

        # instantiate default model and create empty DataFrame for capturing scores
        model = self.BasicModelBuilder(estimator=estimator)
        cv = pd.DataFrame(columns=["Training score", "Validation score"])

        # iterate through each set of features
        for i in np.arange(1, featureSummary.shape[0]):
            top = featureSummary.sort_values("average").index[:-i]
            scores = model_selection.cross_validate(
                estimator=model.model,
                X=data[top],
                y=target,
                cv=nFolds,
                scoring=scoring,
                return_train_score=True,
            )

            # calculate mean scores
            training = scores["train_score"].mean()
            validation = scores["test_score"].mean()

            # append results
            cv.loc[i] = [training, validation]

        # capturing results DataFrame associated with estimator
        cvSummary[estimator] = cv
    return cvSummary

def featureSelectorResultsPlot(self, cvSummary, featureSummary, topSets=0, showFeatures=False):
    """
    Documentation:
        Description:
            For each estimator, visualize the training and validation performance
            for each feature set.
        Parameters:
            cvSummary : dictionary
                Dictionary containing string/Pandas DataFrame key/value pairs, where the
                key is an estimator and the value is a Pandas DataFrame summarizing the
                training and validation performance for each feature set.
            featureSummary : Pandas DataFrame
                Feature importance summary table.
            topSets : int, default = 5
                Number of rows to display of the performance summary table
            showFeatures : boolean, default = False
                Conditional controlling whether to print feature set for best validation
                score.
    """
    for estimator in cvSummary.keys():
        # capture best iteration's feature drop count and performance score
        numDropped = (
            cvSummary[estimator]
            .sort_values(["Validation score"], ascending=False)[:1]
            .index.values[0]
        )
        score = np.round(
            cvSummary[estimator]
            .sort_values(["Validation score"], ascending=False)["Validation score"][:1]
            .values[0],
            5,
        )

        # display performance for the top N feature sets
        if topSets > 0:
            display(cvSummary[estimator].sort_values(["Validation score"], ascending=False)[:topSets])
        if showFeatures:
            featuresUsed = featureSummary.sort_values("average").index[:-numDropped].values
            print(featuresUsed)

        # create multi-line plot
        p = PrettierPlot()
        ax = p.makeCanvas(
            title="{}\nBest validation score = {}, Features dropped = {}".format(
                estimator, score, numDropped
            ),
            xLabel="Features removed",
            yLabel="Accuracy",
            yShift=0.65,
        )

        p.prettyMultiLine(
            x=cvSummary[estimator].index,
            y=["Training score", "Validation score"],
            label=["Training score", "Validation score"],
            df=cvSummary[estimator],
            yUnits="fff",
            markerOn=True,
            ax=ax,
        )
        plt.show()


def featuresUsedSummary(self, cvSummary, featureSummary):
    """
    Documentation:
        Description:
            For each estimator, visualize the training and validation performance
            for each feature set.
        Parameters:
            cvSummary : dictionary
                Dictionary containing string/Pandas DataFrame key/value pairs, where the
                key is an estimator and the value is a Pandas DataFrame summarizing the
                training and validation performance for each feature set.
            featureSummary : Pandas DataFrame
                Feature importance summary table.

    """
    # create empty DataFrame with feature names as index
    df = pd.DataFrame(index=featureSummary.index)

    # iterate through estimators
    for estimator in cvSummary.keys():
        # capture best iteration's feature drop count
        numDropped = (
            cvSummary[estimator]
            .sort_values(["Validation score"], ascending=False)[:1]
            .index.values[0]
        )
        featuresUsed = featureSummary.sort_values("average").index[:-numDropped].values
        columnName = estimator.split(".")[1]

        # create column for estimator and populate with marker
        df[columnName] = np.nan
        df[columnName].loc[featuresUsed] = "X"

    # add counter and fill NaNs
    df["count"] = df.count(axis=1)
    df = df.fillna("")
    return df