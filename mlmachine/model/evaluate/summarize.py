import numpy as np
import pandas as pd


def topBayesOptimModels(self, bayesOptimSummary, numModels=1):
    """
    Documentation:
        Description:
            Aggregate best model(s) for each estimator as determined by Bayesian Optimization
            hyperparamter tuning process.
        Paramaters:
            bayesOptimSummary : Pandas DataFrame
                Pandas DataFrame containing results from Bayesian Optimization process
                execution.
            numModels : int, default = 1
                Number of top models to return per estimator.
        Returns:
            results : dictionary
                Dictionary containing string: list key/value pairs, where the string is the
                name of a algorithm Class and the list contains the integer(s) associated with
                the best model(s) as identified in the hyperparameter optimization summary.
    """

    models = {}
    for estimator in bayesOptimSummary["estimator"].unique():
        estDf = bayesOptimSummary[bayesOptimSummary["estimator"] == estimator].sort_values(
            ["meanLoss","stdLoss","trainTime"], ascending=[True,True,True]
        )["iteration"][:numModels]
        models[estimator] = estDf.values.tolist()
    return models


def regressionStats(model, yTrue, yPred, featureCount, fold=0, dataType='training'):
    """
    Documentation:
        Description:
            Create a dictionary containing information regarding model information and various metrics
            describing the model's performance.
        Paramaters:
            model : model object
                Instantiated model object.
            yTrue : Pandas DataFrame or array
                True labels.
            yPred : Pandas Series or array
                Predicted labels.
            featureCount : int
                Number of features in the observation data. Used to calculate adjusted R-squared.
            fold : int, default = 0
                Indicator for which cross-validation fold the performance is associated with. If 0,
                it is assumed that the evaluation is on an entire dataset (either the full training
                dataset or the full validation dataset), as opposed to a fold.
        Returns:
            results : dictionary
                Dictionary containing string: float key/value pairs, where the string is the

    """
    results = {}

    results['Estimator'] = model.estimator.split(".")[1]
    results['ParameterSet'] = model.modelIter
    results['DataType'] = dataType
    results['Fold'] = fold
    results['N'] = len(yTrue)

    results['ExplainedVariance'] = metrics.explained_variance_score(yTrue, yPred)
    results['MSLE'] = metrics.mean_squared_log_error(yTrue, yPred)
    results['MeanAE'] = metrics.mean_absolute_error(yTrue, yPred)
    results['MedianAE'] = metrics.median_absolute_error(yTrue, yPred)
    results['MSE'] = metrics.mean_squared_error(yTrue, yPred)
    results['RMSE'] = np.sqrt(metrics.mean_squared_error(yTrue, yPred))
    results['R2'] = metrics.r2_score(yTrue, yPred)
    results['AdjustedR2'] = 1 - (1 - metrics.r2_score(yTrue, yPred)) * (len(yTrue) - 1)\
                                    / (len(yTrue) - featureCount - 1)
    return results

def regressionResults(model, XTrain, yTrain, XValid=None, yValid=None, nFolds=3, randomState=1, resultsSummary=None):
    """
    Documentation:
        Description:
            Creates a Pandas DataFrame where each row captures various summary statistics pertaining to a model's performance.
            Captures performance data for training and validation datasets. If no validation set is provided, then
            cross-validation is performed on the training dataset.
        Paramaters:
            model : model object
                Instantiated model object.
            XTrain : Pandas DataFrame
                Training data observations.
            yTrain : Pandas Series
                Training data labels.
            XValid : Pandas DataFrame, default = None
                Validation data observations.
            yValid : Pandas Series, default = None
                Validation data labels.
            nFolds : int, default = 3
                Number of cross-validation folds to use when generating
                CV ROC graph.
            randomState : int, default = 1
                Random number seed.
            resultsSummary : Pndas DataFrame, default = None
                Pandas DataFrame containing various summary statistics pertaining to model performance. If None, returns summary
                Pandas DataFrame for the input model. If resultsSummary DataFrame is provided from a previous run, the new
                performance results are appended to the provivded summary.
        Returns:
            resultsSummary : Pndas DataFrame
                Dataframe containing various summary statistics pertaining to model performance.
    """
    model.fit(XTrain.values, yTrain.values)

    ## training dataset
    yPred = model.predict(XTrain.values)
    results = regressionStats(model=model,
                              yTrue=yTrain.values,
                              yPred=yPred,
                              featureCount=XTrain.shape[1]
                        )
    # create shell results DataFrame and append
    if resultsSummary is None:
        resultsSummary = pd.DataFrame(columns = list(results.keys()))
    resultsSummary = resultsSummary.append(results, ignore_index=True)

    ## Validation dataset
    # if validation data is provided...
    if XValid is not None:
        yPred = model.predict(XValid.values)
        results = regressionStats(model=model,
                                  yTrue=yValid.values,
                                  yPred=yPred,
                                  featureCount=XTrain.shape[1],
                                  dataType='validation'
                            )
        resultsSummary = resultsSummary.append(results, ignore_index=True)
    else:
       # if validation data is not provided, then perform K-fold cross validation on
        # training data
        cv = list(
            model_selection.KFold(
                n_splits=nFolds, shuffle = True, random_state=randomState
            ).split(XTrain, yTrain)
        )

        for i, (trainIx, validIx) in enumerate(cv):
            XTrainCV = XTrain.iloc[trainIx]
            yTrainCV = yTrain.iloc[trainIx]
            XValidCV = XTrain.iloc[validIx]
            yValidCV = yTrain.iloc[validIx]

            yPred = model.fit(XTrainCV.values, yTrainCV.values).predict(XValidCV.values)
            results = self.regressionStats(model=model,
                                      yTrue=yValidCV,
                                      yPred=yPred,
                                      featureCount=XValidCV.shape[1],
                                      dataType='validation',
                                      fold=i+1
                                )
            resultsSummary = resultsSummary.append(results, ignore_index=True)
    return resultsSummary