import ast

import numpy as np
from collections import OrderedDict

from sklearn import model_selection
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, IsolationForest
import sklearn.gaussian_process as gaussian_process
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, LogisticRegression, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import catboost


def oof_generator(self, model, X_train, y_train, X_valid, n_folds=10):
    """
    Documentation:
        Description:
            Generates out-of-fold (OOF) predictions using provided model, training dataset
            and corresponding labels, and a validation dataset.
        Parameters:
            model : sklearn model or pipeline
                Model to be fit.
            X_train : array
                Training data observations.
            y_train : array
                Training target data.
            X_valid : array
                Validation data observations.
            n_folds : int, default=10
                Number of folds for performing cross-validation. Function will generate this
                many sets of out-of-fold predictions.
        Returns:
            oof_train : array
                Array containing observation data to be passed to the meta_learner. oof_train
                is updated throughout the cross-validation process with observations that are
                identified as test observations in each fold.
            oof_valid : array
                Array containing average of all out-of-fold predictions. This is the object to
                pass to the meta_learner.
    """
    # row counts
    n_train = X_machine.shape[0]
    n_valid = X_machine.shape[0]

    # KFold train/test index generator
    kf = KFold(n_splits=n_folds)

    # create shell arrays for holding results
    oof_train = np.zeros((n_train,))
    oof_valid = np.zeros((n_valid,))
    oof_valid_scores = np.empty((n_folds, n_valid))

    # iterate through all KFolds to train model, capture scores
    for i, (train_ix, test_ix) in enumerate(kf.split(X_train)):
        
        # set train/test observations based on KFold indices
        X_train_fold = X_train[train_ix]
        y_train_fold = y_train[train_ix]
        x_test_fold = X_train[test_ix]

        # train model based on training variables and labels
        model.train(X_train_fold, y_train_fold)

        # update segment of oof_train where indices match the indices of the observations
        # used as test observations. these are the "out of fold" observations that we are not
        # considered in the training phase of the model
        oof_train[test_ix] = model.predict(x_test_fold)

        # generate predictions using entire validation dataset, otherwise unused up to this point
        # and capture predictions for each folds
        oof_valid_scores[i, :] = model.predict(X_valid)

    # determine average score of validation predictions
    oof_valid[:] = oof_valid_scores.mean(axis=0)
    return oof_machine.reshape(-1, 1), oof_machine.reshape(-1, 1)


def model_stacker(self, models, bayes_optim_summary, X_train, y_train, X_valid, n_folds, n_jobs):
    """
    Documentation:
        Description:
            Stacking helper function.
        Parameters:
            models : dictionary
                Dictionary of 'x: y' pairs  to be fit.
            bayes_optim_summary : Pandas DataFrame
                Pandas DataFrame containing results from bayesian optimization process.
            X_train : array
                Training data observations.
            y_train : array
                Training target data.
            X_valid : array
                Validation data observations.
            n_folds : int
                Number of folds for performing cross-validation. Function will generate this
                many sets of out-of-fold predictions.
            n_jobs : int
                Number of workers to use when training the model. This parameter will be
                ignored if the model does not have this parameter.
        Returns:
            oof_train : array
                Out-of-fold training data observations.
            oof_valid : array
                Out-of-fold validation data observations.
            columns : list
                List containing estimator names.
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
                estimator_class=estimator,
                model_iter=model_iter,
                n_jobs=n_jobs,
            )

            oof_train_model, oof_valid_model = self.oof_generator(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                n_folds=n_folds,
            )
            try:
                oof_train = np.hstack((oof_train, oof_train_model))
                oof_valid = np.hstack((oof_valid, oof_valid_model))
            except NameError:
                oof_train = oof_train_model
                oof_valid = oof_valid_model
    return oof_train, oof_valid, columns
