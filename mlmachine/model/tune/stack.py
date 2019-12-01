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


def oof_generator(self, model, x_train, y_train, x_valid, n_folds=10):
    """
    documentation:
        description:
            generates out_of_fold (oof) predictions using provided model, training dataset
            and corresponding labels, and a validation dataset.
        parameters:
            model : sklearn model or pipeline
                model to be fit.
            x_train : array
                training dataset.
            y_train : array
                training dataset labels.
            x_valid : array
                validation dataset.
            n_folds : int, default = 10
                number of folds for performing cross_validation. function will generate this
                many sets of out_of_fold predictions.
        returns:
            oof_train : array
                array containing observation data to be passed to the meta_learner. oof_train
                is updated throughout the cross_validation process with observations that are
                identified as test observations in each fold.
            oof_valid : array
                array containing average of all out_of_fold predictions. to be passed to the
                meta_learner.
    """
    # row counts
    n_train = x_train.shape[0]
    n_valid = x_valid.shape[0]

    # kfold train/test index generator
    kf = model_selection.KFold(n_splits=n_folds)

    # create shell arrays for holding results
    oof_train = np.zeros((n_train,))
    oof_valid = np.zeros((n_valid,))
    oof_valid_scores = np.empty((n_folds, n_valid))

    # iterate through all kfolds to train model, capture scores
    for i, (train_ix, test_ix) in enumerate(kf.split(x_train)):
        # set train/test observations based on k_fold indices
        x_train_fold = x_train[train_ix]
        y_train_fold = y_train[train_ix]
        x_test_fold = x_train[test_ix]

        # train model based on training variables and labels
        model.train(x_train_fold, y_train_fold)

        # update segment of oof_train where indices match the indices of the observations
        # used as test observations. these are the "out of fold" observations that we are not
        # considered in the training phase of the model
        oof_train[test_ix] = model.predict(x_test_fold)

        # generate predictions using entire validation dataset, otherwise unused up to this point
        # and capture predictions for each folds
        oof_valid_scores[i, :] = model.predict(x_valid)

    # determine average score of validation predictions
    oof_valid[:] = oof_valid_scores.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_valid.reshape(-1, 1)


def model_stacker(
    self, models, bayes_optim_summary, x_train, y_train, x_valid, n_folds, n_jobs
):
    """
    documentation:
        description:
            stacking helper.
        parameters:
            models : dictionary
                dictionary of 'x : y' pairs  to be fit.
            bayes_optim_summary : pandas DataFrame
                pandas DataFrame containing results from bayesian optimization process
                execution.
            x_train : array
                training dataset.
            y_train : array
                labels for training dataset.
            x_valid : array
                validation dataset.
            n_folkds : int
                number of folds for performing cross_validation. function will generate this
                many sets of out_of_fold predictions.
            n_jobs : int
                number of works to use when training the model. this parameter will be
                ignored if the model does not have this parameter.
        returns:
            oof_train : array
                out_of_fold training data observations.
            oof_valid : array
                out_of_fold validation data observations.
            columns : list
                list containing estimator names.
    """
    columns = []

    # iterate through estimators
    for estimator in models.keys():

        # iterate through parameter set for estimator
        for model_iter in models[estimator]:
            print(estimator + " " + str(model_iter))
            columns.append(estimator + "_" + str(model_iter))

            model = self.BayesOptimModelBuilder(
                bayes_optim_summary=bayes_optim_summary,
                estimator=estimator,
                model_iter=model_iter,
                n_jobs=n_jobs,
            )

            oof_train_model, oof_valid_model = self.oof_generator(
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                n_folds=n_folds,
            )
            try:
                oof_train = np.hstack((oof_train, oof_train_model))
                oof_valid = np.hstack((oof_valid, oof_valid_model))
            except NameError:
                oof_train = oof_train_model
                oof_valid = oof_valid_model
    return oof_train, oof_valid, columns
