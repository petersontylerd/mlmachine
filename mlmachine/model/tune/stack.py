import ast

import numpy as np
from collections import OrderedDict

from sklearn import model_selection
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


def oofGenerator(self, model, XTrain, yTrain, XValid, nFolds=10):
    """
    Documentation:
        Description:
            Generates out-of-fold (oof) predictions using provided model, training dataset
            and corresponding labels, and a validation dataset.
        Parameters:
            model : sklearn model or pipeline
                Model to be fit.
            XTrain : array
                Training dataset.
            yTrain : array
                Training dataset labels.
            XValid : array
                Validation dataset.
            nFolds : int, default = 10
                Number of folds for performing cross-validation. Function will generate this
                many sets of out-of-fold predictions.
        Returns:
            oofTrain : array
                Array containing observation data to be passed to the meta-learner. oofTrain 
                is updated throughout the cross-validation process with observations that are 
                identified as test observations in each fold.
            oofValid : array
                Array containing average of all out-of-fold predictions. To be passed to the
                meta-learner.
    """
    # row counts
    nTrain = XTrain.shape[0]
    nValid = XValid.shape[0]

    # kfold train/test index generator
    kf = model_selection.KFold(n_splits=nFolds)

    # create shell arrays for holding results
    oofTrain = np.zeros((nTrain,))
    oofValid = np.zeros((nValid,))
    oofValidScores = np.empty((nFolds, nValid))

    # iterate through all kfolds to train model, capture scores
    for i, (trainIx, testIx) in enumerate(kf.split(XTrain)):
        # set train/test observations based on KFold indices
        XTrainFold = XTrain[trainIx]
        yTrainFold = yTrain[trainIx]
        XTestFold = XTrain[testIx]

        # train model based on training variables and labels
        model.train(XTrainFold, yTrainFold)

        # update segment of oofTrain where indices match the indices of the observations
        # used as test observations. These are the "out of fold" observations that we are not
        # considered in the training phase of the model
        oofTrain[testIx] = model.predict(XTestFold)

        # generate predictions using entire validation dataset, otherwise unused up to this point
        # and capture predictions for each folds
        oofValidScores[i, :] = model.predict(XValid)

    # determine average score of validation predictions
    oofValid[:] = oofValidScores.mean(axis=0)
    return oofTrain.reshape(-1, 1), oofValid.reshape(-1, 1)


def modelStacker(self, models, bayesOptimSummary, XTrain, yTrain, XValid, nFolds, nJobs):
    """
    Documentation:
        Description:
            Stacking helper.
        Parameters:
            models : dictionary
                Dictionary of 'X : y' pairs  to be fit.
            bayesOptimSummary : Pandas DataFrame
                Pandas DataFrame containing results from Bayesian Optimization process 
                execution. 
            XTrain : array
                Training dataset.
            yTrain : array
                Labels for training dataset.
            XValid : array
                Validation dataset.
            nFolkds : int
                Number of folds for performing cross-validation. Function will generate this
                many sets of out-of-fold predictions.
            nJobs : int
                Number of works to use when training the model. This parameter will be 
                ignored if the model does not have this parameter.
        Returns:
            oofTrain : array
                Out-of-fold training data observations.
            oofValid : array
                Out-of-fold validation data observations.
            columns : list
                List containing estimator names.
    """
    columns = []
    
    # iterate through estimators
    for estimator in models.keys():
    
        # iterate through parameter set for estimator
        for modelIter in models[estimator]:
            print(estimator + " " + str(modelIter))
            columns.append(estimator + "_" + str(modelIter))
            
            model = self.BayesOptimModelBuilder(
                bayesOptimSummary=bayesOptimSummary,
                estimator=estimator,
                modelIter=modelIter,
                nJobs=nJobs,
            )

            oofTrainModel, oofValidModel = self.oofGenerator(
                model=model, XTrain=XTrain, yTrain=yTrain, XValid=XValid, nFolds=nFolds
            )
            try:
                oofTrain = np.hstack((oofTrain, oofTrainModel))
                oofValid = np.hstack((oofValid, oofValidModel))
            except NameError:
                oofTrain = oofTrainModel
                oofValid = oofValidModel
    return oofTrain, oofValid, columns
