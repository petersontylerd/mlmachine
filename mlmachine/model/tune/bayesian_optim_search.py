import ast
import csv
import inspect
import os
import pickle
from time import gmtime, strftime
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

import numpy as np
import pandas as pd

from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from hyperopt.pyll.stochastic import sample


from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    IsolationForest,
)
from sklearn.linear_model import (
    Lasso,
    Ridge,
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    SGDRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import (
    KFold,
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.base import clone

from sklearn.metrics import get_scorer

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import catboost

from prettierplot.plotter import PrettierPlot
from prettierplot import style


def objective(space, results_file, estimator_class, training_features, training_target, validation_features, validation_target, scoring, n_folds, n_jobs, result_dir):
    """
    Documentation:

        ---
        Description:
            Customizable objective function to minimize in the bayesian hyper-parameter optimization
            process.

        ---
        Parameters:
            space : dictionary
                Dictionary containg 'parameter: value distribution' key/value pairs. The key specifies
                the parameter of the esto,atpr and the optimization process draws trial values from
                that distribution.
            results_file : str
                File location of results summary csv.
            estimator_class : str or sklearn api object
                The model to be fit.
            training_features : array
                Training data observations.
            training_target : array
                Training target data.
            validation_features : array
                Validation data observations.
            validation_target : array
                Validation target data.
            scoring : str
                Evaluation method for scoring model performance. The following metrics are supported:
                    - "accuracy"
                    - "f1_macro"
                    - "f1_micro"
                    - "neg_mean_squared_error"
                    - "roc_auc"
                    - "root_mean_squared_error"
                    - "root_mean_squared_log_error"
                Note that "root_mean_squared_log_error" is not implemented in sklearn. If
                "root_mean_squared_log_error" is specified, model is optimized using
                "neg_mean_squared_error" and then the square root is taken of the absolute
                value of the results, effectively creating the "root_mean_squared_log_error"
                score to minimize.
            n_folds : int
                Number of folds for cross-validation.
            n_jobs : int
                Number of workers to deploy upon execution, if applicable.

        ---
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
        score_transform = "rmse"
    elif scoring == "root_mean_squared_log_error":
        scoring = "neg_mean_squared_log_error"
        score_transform = "rmsle"
    else:
        score_transform = scoring

    # create model, estimator name to be used as column header in DataFrame
    model, estimator_name = model_type_check(estimator=estimator_class, n_jobs=n_jobs, params=space)

    # execute cross-validation scoring
    cv = cross_val_score(
        estimator=model.custom_model,
        X=training_features,
        y=training_target,
        verbose=False,
        n_jobs=n_jobs,
        cv=n_folds,
        scoring=scoring,
    )

    # validation score
    model.custom_model.fit(training_features, training_target)
    validation_scorer = get_scorer(scoring)
    validation_score = validation_scorer(model.custom_model, validation_features, validation_target)

    # log runtime
    run_time = timer() - start

    # calculate loss based on scoring method
    if score_transform in ["accuracy", "f1_macro", "f1_micro", "precision", "recall", "roc_auc"]:
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

    ## load results summary and extract best loss value for current estimator
    df = pd.read_csv(results_file)
    df = df[df["estimator"] == estimator_name]

    if df.empty:
        best_loss = np.inf
    else:
        best_loss = df["loss"].min()

    # compare current loss to best loss and pickle model if improved
    if best_loss > loss:

        #
        with open(os.path.join(result_dir, "{}.pkl".format(estimator_name)), 'wb') as handle:
            pickle.dump(model.custom_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # export results to csv
    with open(results_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                ITERATION,
                estimator_name,
                scoring,
                validation_score,
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

    # return result for optimizer
    return {
        "iteration": ITERATION,
        "estimator": estimator_name,
        "scoring": scoring,
        "validation_score": validation_score,
        "loss": loss,
        "mean_score": mean_score,
        "std_score": std_score,
        "min_score": min_score,
        "max_score": max_score,
        "train_time": run_time,
        "status": STATUS_OK,
        "params": space,
    }

def exec_bayes_optim_search(self, estimator_parameter_space, training_features, training_target, validation_features, validation_target,
                            scoring, columns=None, n_folds=3, n_jobs=4, iters=50, show_progressbar=False):
    """
    Documentation:

        ---
        Definition:
            Perform bayesian hyper-parameter optimization across a set of models and parameter value
            distributions.

        ---
        Parameters:
            estimator_parameter_space : dictionary of dictionaries
                Dictionary of nested dictionaries. Outer key is a model, and the corresponding value is
                a dictionary. Each nested dictionary contains 'parameter: value distribution' key/value
                pairs. The inner dictionary key specifies the parameter of the model to be tuned, and the
                value is a distribution of values from which trial values are drawn.
            training_features : array
                Training data observations.
            training_target : array
                Training target data.
            validation_features : array
                Validation data observations.
            validation_target : array
                Validation target data.
            scoring : str
                Evaluation method for scoring model performance. The following metrics are supported:
                    - "accuracy"
                    - "f1_macro"
                    - "f1_micro"
                    - "neg_mean_squared_error"
                    - "roc_auc"
                    - "root_mean_squared_error"
                    - "root_mean_squared_log_error"
                Note that "root_mean_squared_log_error" is not implemented in sklearn. If
                "root_mean_squared_log_error" is specified, model is optimized using
                "neg_mean_squared_error" and then the square root is taken of the absolute
                value of the results, effectively creating the "root_mean_squared_log_error"
                score to minimize.
            columns : dict or list, default=None
                If list of feature names is provided, all estimators use the same feature set. If dictionary
                provided, dictionary should contain 'estimator: feature names' key/value pairs. The estimator
                key is a string representing an estimator class, and the associated list is the list of features
                to use with that estimator. The dictionary-approach enables utilization of different features
                sets for each estimator.
            n_folds : int, default=3
                Number of folds for cross-validation.
            n_jobs : int, default - 4
                Number of workers to deploy upon execution, if applicable.
            iters : int, default=50
                Number of iterations to run process.
            show_progressbar : boolean, default=False
                Controls whether to print progress bar to console during training.
    """
    # set results logging destination
    results_file = os.path.join(
        self.training_object_dir,
        "bayes_optimization_summary.csv"
    )

    # add file header
    with open(results_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            [
                "iteration",
                "estimator",
                "scoring",
                "validation_score",
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

    # iterate through each estimator
    for estimator_class in estimator_parameter_space.keys():

        # set iteration count to zero
        global ITERATION
        ITERATION = 0

        # establish feature space for hyper-parameter search for current estimator
        space = estimator_parameter_space[estimator_class]

        ## training data
        # conditionally handle input training data
        if isinstance(training_features, pd.core.frame.DataFrame):

            # if column subset is provided as a dictionary
            if isinstance(columns, dict):
                try:
                    input_training_features = training_features[columns[estimator_class]]
                except KeyError:
                    input_training_features = training_features.copy()

            # if column subset is provided as a list
            if isinstance(columns, list):
                try:
                    input_training_features = training_features[columns]
                except KeyError:
                    input_training_features = training_features.copy()

            elif columns is None:
                pass

            # use underlying numpy ndarray
            input_training_features = training_features.values
        elif isinstance(data, np.ndarray):

            # create a copy of the underlying data array
            input_training_features = training_features.copy()
        else:
            raise AttributeError(
                "input data set must be either a Pandas DataFrame or a numpy ndarray"
            )

        # conditionally handle input training target
        if isinstance(training_target, pd.core.series.Series):

            # use underlying numpy ndarray
            input_training_target = training_target.values
        elif isinstance(training_target, np.ndarray):

            # create a copy of the underlying data array
            input_training_target = training_target.copy()
        else:
            raise AttributeError(
                "input target must be either a Pandas Series or a numpy ndarray"
            )

        ## validation data
        # conditionally handle input validation data
        if isinstance(validation_features, pd.core.frame.DataFrame):

            # if column subset is provided as a dictionary
            if isinstance(columns, dict):
                try:
                    input_validation_features = validation_features[columns[estimator_class]]
                except KeyError:
                    input_validation_features = validation_features.copy()

            # if column subset is provided as a list
            if isinstance(columns, list):
                try:
                    input_validation_features = validation_features[columns]
                except KeyError:
                    input_validation_features = validation_features.copy()

            elif columns is None:
                pass

            # use underlying numpy ndarray
            input_validation_features = validation_features.values
        elif isinstance(data, np.ndarray):

            # create a copy of the underlying data array
            input_validation_features = validation_features.copy()
        else:
            raise AttributeError(
                "input data set must be either a Pandas DataFrame or a numpy ndarray"
            )

        # conditionally handle input validation target
        if isinstance(validation_target, pd.core.series.Series):

            # use underlying numpy ndarray
            input_validation_target = validation_target.values
        elif isinstance(validation_target, np.ndarray):

            # create a copy of the underlying data array
            input_validation_target = validation_target.copy()
        else:
            raise AttributeError(
                "input target must be either a Pandas Series or a numpy ndarray"
            )

        # override default arguments with next estimator_class and the cv parameters
        objective.__defaults__ = (
            results_file,
            estimator_class,
            input_training_features,
            input_training_target,
            input_validation_features,
            input_validation_target,
            scoring,
            n_folds,
            n_jobs,
            self.training_models_object_dir,
        )

        # run optimization
        if show_progressbar:
            print("\n" + "#" * 100)
            print(f"\nTuning {estimator_class}\n")

        # minimize the objective function
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=iters,
            trials=Trials(),
            show_progressbar=show_progressbar,
        )

    #
    self.bayes_optim_summary = pd.read_csv(
                                    os.path.join(
                                        self.training_object_dir,
                                        "bayes_optimization_summary.csv"
                                    ),
                                    na_values="nan"
                                )

class BayesOptimClassifierBuilder(ClassifierMixin):
    """
    Documentation:

        ---
        Description:
            Helper class for instantiating a classifier model with a provided parameter set as stored
            on the bayesian optimization log file.

        ---
        Parameters:
            bayes_optim_summary : Pandas DataFrame
                Pandas DataFrame containing results from bayesian optimization process.
            estimator_class : str or sklearn api object
                Name of estimator to build.
            model_iter : int
                Numeric identifier for a specific parameter set used in a training
                iteration.
            n_jobs : int, default=4
                Number of workers to use when training the model. This parameter will be
                ignored if the model does not have this parameter.

        ---
        Returns:
            model : model object
                Model instantiated using parameter set.
    """

    def __init__(self, bayes_optim_summary, estimator_class, model_iter, n_jobs=4):
        self.bayes_optim_summary = bayes_optim_summary
        self.estimator_class = estimator_class
        self.model_iter = model_iter
        self.n_jobs = n_jobs
        self.params = self.bayes_optim_summary[
            (self.bayes_optim_summary["estimator"] == self.estimator_class)
            & (self.bayes_optim_summary["iteration"] == self.model_iter)
        ]["params"].values[0]

        # turn string representation of dictionary into dictionary
        self.params = ast.literal_eval(self.params)

        # convert estimator argument to sklearn api object if needed
        if isinstance(self.estimator_class, str):
            self.estimator_name = self.estimator_class
            self.estimator_class = eval(self.estimator_class)

        # capture available model arguments and set probabily and n_jobs where applicable
        estimator_args = inspect.signature(self.estimator_class.__init__).parameters.keys()

        # add probably=True if estimator accept the argument
        if "probability" in estimator_args:
            self.params["probability"] = True

        # specify n_jobs variable if estimator accept the argument
        if "n_jobs" in estimator_args:
            self.params["n_jobs"] = self.n_jobs

        # # specify random_state variable if estimator accept the argument
        # if "random_state" in estimator_args:
        #     self.params["random_state"] = self.random_state

        # special handling for XGBoost estimators (suppresses warning messages)
        if "XGB" in self.estimator_name:
            self.params["verbosity"] = 0
            self.params["use_label_encoder"] = False

        # instantiate model
        self.custom_model = self.estimator_class(**self.params)

    def train(self, X_train, y_train):
        self.custom_model.fit(X_train, y_train)

    def predict(self, x):
        return self.custom_model.predict(x)

    def predict_proba(self, x):
        return self.custom_model.predict_proba(x)

    def fit(self, x, y):
        return self.custom_model.fit(x, y)

    def feature_importances_(self, x, y):
        return self.custom_model.fit(x, y).feature_importances_

class BayesOptimRegressorBuilder(RegressorMixin):
    """
    Documentation:

        ---
        Description:
            Helper class for instantiating a regression model with a provided parameter set as stored
            on the bayesian optimization log file.

        ---
        Parameters:
            bayes_optim_summary : Pandas DataFrame
                Pandas DataFrame containing results from bayesian optimization process.
            estimator_class : str or sklearn api object
                Name of estimator to build.
            model_iter : int
                Numeric identifier for a specific parameter set used in a training
                iteration.
            n_jobs : int, default=4
                Number of workers to use when training the model. This parameter will be
                ignored if the model does not have this parameter.

        ---
        Returns:
            model : model object
                Model instantiated using parameter set.
    """

    def __init__(self, bayes_optim_summary, estimator_class, model_iter, n_jobs=4):
        self.bayes_optim_summary = bayes_optim_summary
        self.estimator_class = estimator_class
        self.model_iter = model_iter
        self.n_jobs = n_jobs
        self.params = self.bayes_optim_summary[
            (self.bayes_optim_summary["estimator"] == self.estimator_class)
            & (self.bayes_optim_summary["iteration"] == self.model_iter)
        ]["params"].values[0]

        # turn string representation of dictionary into dictionary
        self.params = ast.literal_eval(self.params)

        # convert estimator argument to sklearn api object if needed
        if isinstance(self.estimator_class, str):
            self.estimator_name = self.estimator_class
            self.estimator_class = eval(self.estimator_class)

        # capture available model arguments and set probabily and n_jobs where applicable
        estimator_args = inspect.signature(self.estimator_class.__init__).parameters.keys()

        # add probably=True if estimator accept the argument
        if "probability" in estimator_args:
            self.params["probability"] = True

        # specify n_jobs variable if estimator accept the argument
        if "n_jobs" in estimator_args:
            self.params["n_jobs"] = self.n_jobs

        # specify random_state variable if estimator accept the argument
        if "random_state" in estimator_args:
            self.params["random_state"] = self.random_state

        # special handling for XGBoost estimators (suppresses warning messages)
        if "XGB" in self.estimator_name:
            self.params["verbosity"] = 0
            self.params["use_label_encoder"] = False

        # instantiate model
        self.custom_model = self.estimator_class(**self.params)

    def train(self, X_train, y_train):
        self.custom_model.fit(X_train, y_train)

    def predict(self, x):
        return self.custom_model.predict(x)

    def predict_proba(self, x):
        return self.custom_model.predict_proba(x)

    def fit(self, x, y):
        return self.custom_model.fit(x, y)

    def feature_importances_(self, x, y):
        return self.custom_model.fit(x, y).feature_importances_

class BayesOptimModelBuilder(BaseEstimator):
    """
    Documentation:

        ---
        Description:
            Helper class for instantiating a generic model with a provided parameter set as stored
            on the bayesian optimization log file.

        ---
        Parameters:
            bayes_optim_summary : Pandas DataFrame
                Pandas DataFrame containing results from bayesian optimization process.
            estimator_class : str or sklearn api object
                Name of estimator to build.
            model_iter : int
                Numeric identifier for a specific parameter set used in a training
                iteration.
            n_jobs : int, default=4
                Number of workers to use when training the model. This parameter will be
                ignored if the model does not have this parameter.

        ---
        Returns:
            model : model object
                Model instantiated using parameter set.
    """

    def __init__(self, bayes_optim_summary, estimator_class, model_iter, n_jobs=4):
        self.bayes_optim_summary = bayes_optim_summary
        self.estimator_class = estimator_class
        self.model_iter = model_iter
        self.n_jobs = n_jobs
        self.params = self.bayes_optim_summary[
            (self.bayes_optim_summary["estimator"] == self.estimator_class)
            & (self.bayes_optim_summary["iteration"] == self.model_iter)
        ]["params"].values[0]
        self._estimator_type = "classifier"


        # turn string representation of dictionary into dictionary
        self.params = ast.literal_eval(self.params)

        # convert estimator argument to sklearn api object if needed
        if isinstance(self.estimator_class, str):
            self.estimator_name = self.estimator_class
            self.estimator_class = eval(self.estimator_class)

        # capture available model arguments and set probabily and n_jobs where applicable
        estimator_args = inspect.signature(self.estimator_class.__init__).parameters.keys()

        # add probably=True if estimator accept the argument
        if "probability" in estimator_args:
            self.params["probability"] = True

        # specify n_jobs variable if estimator accept the argument
        if "n_jobs" in estimator_args:
            self.params["n_jobs"] = self.n_jobs

        # specify random_state variable if estimator accept the argument
        if "random_state" in estimator_args:
            self.params["random_state"] = self.random_state

        # special handling for XGBoost estimators (suppresses warning messages)
        if "XGB" in self.estimator_name:
            self.params["verbosity"] = 0
            self.params["use_label_encoder"] = False

        # instantiate model
        self.custom_model = self.estimator_class(**self.params)

    def train(self, X_train, y_train):
        self.custom_model.fit(X_train, y_train)

    def predict(self, x):
        return self.custom_model.predict(x)

    def predict_proba(self, x):
        return self.custom_model.predict_proba(x)

    def fit(self, x, y):
        return self.custom_model.fit(x, y)

    def feature_importances_(self, x, y):
        return self.custom_model.fit(x, y).feature_importances_

class BasicRegressorBuilder(RegressorMixin):
    """
    Documentation:

        ---
        Description:
            Helper class for instantiating a regressor model with a provided parameter set.

        ---
        Parameters:
            estimator_class : str or sklearn api object
                Model to instantiate.
            params : dictionary, default=None
                Dictionary containing 'parameter: value' pairs. If no dictionary is provided,
                then an empty dictionary is created by default, which instantiates the model with
                its default parameter values.
            n_jobs : int, default=4
                Number of workers to use when training the model. This parameter will be
                ignored if the model does not have this parameter.

        ---
        Returns:
            model : model object
                Model instantiated using parameter set.
    """

    def __init__(self, estimator_class, params=None, n_jobs=4, random_state=0):

        self.estimator_class = estimator_class
        self.params = {} if params is None else params
        self.n_jobs = n_jobs
        self.random_state = random_state

        # convert estimator argument to sklearn api object if needed
        if isinstance(self.estimator_class, str):
            self.estimator_name = self.estimator_class
            self.estimator_class = eval(self.estimator_class)

        # capture available model arguments and set probabily and n_jobs where applicable
        estimator_args = inspect.signature(self.estimator_class.__init__).parameters.keys()

        # add probably=True if estimator accept the argument
        if "probability" in estimator_args:
            self.params["probability"] = True

        # specify n_jobs variable if estimator accept the argument
        if "n_jobs" in estimator_args:
            self.params["n_jobs"] = self.n_jobs

        # specify random_state variable if estimator accept the argument
        if "random_state" in estimator_args:
            self.params["random_state"] = self.random_state

        # special handling for XGBoost estimators (suppresses warning messages)
        if "XGB" in self.estimator_name:
            self.params["verbosity"] = 0
            self.params["use_label_encoder"] = False

        # instantiate model
        self.custom_model = self.estimator_class(**self.params)

    def train(self, X_train, y_train):
        self.custom_model.fit(X_train, y_train)

    def predict(self, x):
        return self.custom_model.predict(x)

    def predict_proba(self, x):
        return self.custom_model.predict_proba(x)

    def fit(self, x, y):
        return self.custom_model.fit(x, y)

    def feature_importances_(self, x, y):
        return self.custom_model.fit(x, y).feature_importances_

class BasicClassifierBuilder(ClassifierMixin):
    """
    Documentation:

        ---
        Description:
            Helper class for instantiating a classifier model with a provided parameter set.

        ---
        Parameters:
            estimator_class : str or sklearn api object
                model to instantiate.
            params : dictionary, default=None
                dictionary containing 'parameter : value' pairs. If no dictionary is provided,
                then an empty dictionary is created by default, which instantiates the model with
                its default parameter values.
            n_jobs : int, default=4
                Number of workers to use when training the model. This parameter will be
                ignored if the model does not have this parameter.

        ---
        Returns:
            model : model object
                model instantiated using parameter set. model possesses train, predict, fit
                and feature_importances methods.
    """

    def __init__(self, estimator_class, params=None, n_jobs=4, random_state=0):

        self.estimator_class = estimator_class
        self.params = {} if params is None else params
        self.n_jobs = n_jobs
        self.random_state = random_state

        # convert estimator argument to sklearn api object if needed
        if isinstance(self.estimator_class, str):
            self.estimator_name = self.estimator_class
            self.estimator_class = eval(self.estimator_class)

        # capture available model arguments and set probabily and n_jobs where applicable
        estimator_args = inspect.signature(self.estimator_class.__init__).parameters.keys()

        # add probably=True if estimator accept the argument
        if "probability" in estimator_args:
            self.params["probability"] = True

        # specify n_jobs variable if estimator accept the argument
        if "n_jobs" in estimator_args:
            self.params["n_jobs"] = self.n_jobs

        # specify random_state variable if estimator accept the argument
        if "random_state" in estimator_args:
            self.params["random_state"] = self.random_state

        # special handling for XGBoost estimators (suppresses warning messages)
        if "XGB" in self.estimator_name:
            self.params["verbosity"] = 0
            self.params["use_label_encoder"] = False

        # instantiate model
        self.custom_model = self.estimator_class(**self.params)

    def train(self, X_train, y_train):
        self.custom_model.fit(X_train, y_train)

    def predict(self, x):
        return self.custom_model.predict(x)

    def predict_proba(self, x):
        return self.custom_model.predict_proba(x)

    def fit(self, x, y):
        return self.custom_model.fit(x, y)

    def feature_importances_(self, x, y):
        return self.custom_model.fit(x, y).feature_importances_

class BasicModelBuilder(BaseEstimator):
    """
    Documentation:

        ---
        Description:
            Helper class for instantiating a generic model with a provided parameter set.

        ---
        Parameters:
            estimator_class : str or sklearn api object
                Model to instantiate.
            params : dictionary, default=None
                Dictionary containing 'parameter : value' pairs. If no dictionary is provided,
                then an empty dictionary is created by default, which instantiates the model with
                its default parameter values.
            n_jobs : int, default=4
                Number of workers to use when training the model. This parameter will be
                ignored if the model does not have this parameter.

        ---
        Returns:
            model : model object
                Model instantiated using parameter set.
    """

    def __init__(self, estimator_class, params=None, n_jobs=4, random_state=0):

        self.estimator_class = estimator_class
        self.params = {} if params is None else params
        self.n_jobs = n_jobs
        self.random_state = random_state

        # convert estimator argument to sklearn api object if estimator passed as str
        if isinstance(self.estimator_class, str):
            self.estimator_name = self.estimator_class
            self.estimator_class = eval(self.estimator_class)
        else:
            self.estimator_name = self.estimator_class.__name__

        # capture available model arguments and set probabily and n_jobs where applicable
        estimator_args = inspect.signature(self.estimator_class.__init__).parameters.keys()

        # add probably=True if estimator accept the argument
        if "probability" in estimator_args:
            self.params["probability"] = True

        # specify n_jobs variable if estimator accept the argument
        if "n_jobs" in estimator_args:
            self.params["n_jobs"] = self.n_jobs

        # specify random_state variable if estimator accept the argument
        if "random_state" in estimator_args:
            self.params["random_state"] = self.random_state

        # special handling for XGBoost estimators (suppresses warning messages)
        if "XGB" in self.estimator_name:
            self.params["verbosity"] = 0
            self.params["use_label_encoder"] = False

        # instantiate model
        self.custom_model = self.estimator_class(**self.params)

    def train(self, X_train, y_train):
        self.custom_model.fit(X_train, y_train)

    def predict(self, x):
        return self.custom_model.predict(x)

    def predict_proba(self, x):
        return self.custom_model.predict_proba(x)

    def fit(self, x, y):
        return self.custom_model.fit(x, y)

    def feature_importances_(self, x, y):
        return self.custom_model.fit(x, y).feature_importances_

def unpack_bayes_optim_summary(self, bayes_optim_summary, estimator_class):
    """
    Documentation:

        ---
        Definition:
            Unpack bayesian optimization results summary data for a specified estimator
            into a Pandas DataFrame where there is one column for each parameter.

        ---
        Parameters:
            bayes_optim_summary : Pandas DataFrame
                Pandas DataFrame containing results from bayesian optimization process.
            estimator_class : str or sklearn api object
                Name of estimator to build.

        ---
        Returns:
            estimator_param_summary : Pandas DataFrame
                Pandas DataFrame where each row is a record of the parameters used and the
                loss recorded in an iteration.
    """
    # subset bayes_optim_summary to specified estimator_class
    estimator_df = bayes_optim_summary[
        bayes_optim_summary["estimator"] == estimator_class
    ].reset_index(drop=True)

    # create a new dataframe for storing estimator_class's parameters
    estimator_summary = pd.DataFrame(
        columns=list(ast.literal_eval(estimator_df.loc[0, "params"]).keys()),
        index=list(range(len(estimator_df))),
    )

    # append parameter set to estimator_summary
    for i, params in enumerate(estimator_df["params"]):
        estimator_summary.loc[i, :] = list(ast.literal_eval(params).values())

    # add columns for loss and iteration number
    estimator_summary["iter_loss"] = estimator_df["loss"]
    estimator_summary["iteration"] = estimator_df["iteration"]

    return estimator_summary

def model_loss_plot(self, bayes_optim_summary, estimator_class, chart_scale=15, trim_outliers=True, outlier_control=1.5,
                    title_scale=0.7, color_map="viridis", save_plots=False):
    """
    Documentation:

        ---
        Definition:
            Visualize how the bayesian optimization loss changes over time across all iterations.
            Extremely poor results are removed from visualized dataset by two filters.
                1) Loss values worse than [loss mean + (2 x loss standard deviation)]
                2) Loss values worse than [median * outliers_control]. 'outlier_control' is a parameter
                   that can be set during function execution.

        ---
        Parameters:
            bayes_optim_summary : Pandas DataFrame
                Pandas DataFrame containing results from bayesian optimization process.
            estimator_class : str or sklearn api object
                Name of estimator to visualize.
            chart_scale : float, default=15
                Control chart proportions. Higher values scale up size of chart objects, lower
                values scale down size of chart objects.
            trim_outliers : boolean, default=True
                Remove extremely high (poor) results by trimming values where the loss is greater
                than 2 standard deviations away from the mean.
            outlier_control : float: default=1.5
                Controls enforcement of outlier trimming. Value is multiplied by median, and the resulting
                product is the cap placed on loss values. Values higher than this cap will be excluded.
                Lower values of outlier_control apply more extreme filtering to loss values.
            title_scale : float, default=0.7
                Controls the scaling up (higher value) and scaling down (lower value) of the size of
                the main chart title, the x_axis title and the y_axis title.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            save_plots : boolean, default = False
                Controls whether model loss plot imgaes are saved to the experiment directory.
    """
    # unpack bayes_optim_summary parameters for an estimator_class
    estimator_summary = self.unpack_bayes_optim_summary(
        bayes_optim_summary=bayes_optim_summary, estimator_class=estimator_class
    )

    # apply outlier trimming
    if trim_outliers:
        mean = estimator_summary["iter_loss"].mean()
        median = estimator_summary["iter_loss"].median()
        std = estimator_summary["iter_loss"].std()
        cap = mean + (2.0 * std)
        estimator_summary = estimator_summary[
            (estimator_summary["iter_loss"] < cap)
            & (estimator_summary["iter_loss"] < outlier_control * median)
        ]

    # create color list based on color_map
    color_list = style.color_gen(name=color_map, num=3)

    # create prettierplot object
    p = PrettierPlot(chart_scale=chart_scale)

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title=f"Loss by iteration - {estimator_class}",
        y_shift=0.8,
        position=111,
        title_scale=title_scale,
    )

    # add regression plot to canvas
    p.reg_plot(
        x="iteration",
        y="iter_loss",
        data=estimator_summary,
        y_units="ffff",
        line_color=color_list[0],
        dot_color=color_list[1],
        alpha=0.6,
        line_width=0.4,
        dot_size=10.0,
        ax=ax,
    )

    # save plots or show
    if save_plots:
        plot_path = os.path.join(
            self.training_plots_model_loss_dir,
            f"{estimator_class}.jpg",
        )
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()

def model_param_plot(self, bayes_optim_summary, estimator_class, estimator_parameter_space, n_iter, chart_scale=15,
                    color_map="viridis", title_scale=1.2, show_single_str_params=False, save_plots=False):
    """
    Documentation:

        ---
        Definition:
            Visualize hyperparameter optimization over all iterations. Compares theoretical distribution to
            the distribution of values that were actually chosen, and visualizes how parameter value
            selections changes over time.

        ---
        Parameters:
            bayes_optim_summary : Pandas DataFrame
                Pandas DataFrame containing results from bayesian optimization process.
            estimator_class : str or sklearn api object
                Name of estimator to visualize.
            estimator_parameter_space : dictionary of dictionaries
                Dictionary of nested dictionaries. Outer key is an estimator, and the corresponding value is
                a dictionary. Each nested dictionary contains 'parameter: value distribution' key/value
                pairs. The inner dictionary key specifies the parameter of the model to be tuned, and the
                value is a distribution of values from which trial values are drawn.
            n_iter : int
                Number of iterations to draw from theoretical distribution in order to visualize the
                theoretical distribution. Higher number leader to more robust distribution but can take
                considerably longer to create.
            chart_scale : float, default=15
                Controls proportions of visualizations. larger values scale visual up in size, smaller values
                scale visual down in size.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            title_scale : float, default=1.2
                Controls the scaling up (higher value) and scaling down (lower value) of the size of
                the main chart title, the x_axis title and the y_axis title.
            show_single_str_params : boolean, default=False
                Controls whether to display visuals for string attributes where there is only one unique value,
                i.e. there was only one choice for the optimization procedure to choose from during each iteration.
            save_plots : boolean, default=False
                Controls whether model loss plot imgaes are saved to the experiment directory.
    """
    # unpack bayes_optim_summary parameters for an estimator_class
    estimator_summary = self.unpack_bayes_optim_summary(
        bayes_optim_summary=bayes_optim_summary, estimator_class=estimator_class
    )

    # override None with string representation
    estimator_summary = estimator_summary.replace([None], "None")

    # subset estimator_parameter_space to space for the specified estimator_class
    estimator_space = estimator_parameter_space[estimator_class]

    if not save_plots:
        print("*" * 100)
        print("* {}".format(estimator_class))
        print("*" * 100)

    # iterate through each parameter
    for param in estimator_space.keys():

        # sample from theoretical distribution for n_iters
        theoretical_dist = []
        for _ in range(n_iter):
            theoretical_dist.append(sample(estimator_space)[param])

        ## override None with string representation
        # theoretical distribution
        theoretical_dist = ["none" if v is None else v for v in theoretical_dist]
        theoretical_dist = np.array(theoretical_dist)

        # actual distribution
        actual_dist = estimator_summary[param].tolist()
        actual_dist = ["none" if v is None else v for v in actual_dist]
        actual_dist = np.array(actual_dist)

        # limit estimator_summary to "iteration" and current "param" columns
        actual_iter_df = estimator_summary[["iteration", param]]

        # identify how many values in param column are zero or one
        zeros_and_ones = (actual_iter_df[param].eq(True) | actual_iter_df[param].eq(False)).sum()

        # param column only contains zeros and ones, store string representations of "TRUE" and "FALSE"
        if zeros_and_ones == actual_iter_df.shape[0]:
            actual_iter_df = actual_iter_df.replace({True: "TRUE", False: "FALSE"})

        # if theoreitcal distribution has dtype -- np.bool_, store string representations of "TRUE" and "FALSE"
        if isinstance(theoretical_dist[0], np.bool_):
            theoretical_dist = np.array(["TRUE" if i == True else "FALSE" for i in theoretical_dist.tolist()])

            estimator_summary = estimator_summary.replace([True], "TRUE")
            estimator_summary = estimator_summary.replace([False], "FALSE")

        # if theoretical distribution contains str data, then treat this as an object/category parameter
        if any(isinstance(d, str) for d in theoretical_dist):

            # generate color list for stripplot
            stripplot_color_list = style.color_gen(name=color_map, num=len(actual_iter_df[param].unique()) + 1)

            # generate color list for bar chart
            bar_color_list = style.color_gen(name=color_map, num=3)

            # identify unique values and associated count in theoretical distribution
            unique_vals_theo, unique_counts_theo = np.unique(theoretical_dist, return_counts=True)

            # if theoretical distribution only has one unique value and show_single_str_params is set to True
            if len(unique_vals_theo) > 1 or show_single_str_params:

                # identify unique values and associated count in actual distribution
                unique_vals_actual, unique_counts_actual = np.unique(actual_dist, return_counts=True)

                # store data in DataFrame
                df = pd.DataFrame({"param": unique_vals_actual, "Theorical": unique_counts_theo, "Actual": unique_counts_actual})

                # create prettierplot object
                p = PrettierPlot(chart_scale=chart_scale, plot_orientation = "wide_narrow")

                # add canvas to prettierplot object
                ax = p.make_canvas(
                    title=f"Selection vs. theoretical distribution\n* {estimator_class} - {param}",
                    y_shift=0.8,
                    position=121,
                    title_scale=title_scale,
                )

                # add faceted bar chart to canvas
                p.facet_cat(
                    df=df,
                    feature="param",
                    color_map=bar_color_list[:-1],
                    bbox=(1.0, 1.15),
                    alpha=1.0,
                    legend_labels=df.columns[1:].values,
                    x_units=None,
                    ax=ax,
                )

                # add canvas to prettierplot object
                ax = p.make_canvas(
                    title=f"Selection by iteration\n* {estimator_class} - {param}",
                    y_shift=0.5,
                    position=122,
                    title_scale=title_scale,
                )

                # add stripply to canvas
                sns.stripplot(
                    x="iteration",
                    y=param,
                    data=estimator_summary,
                    jitter=0.3,
                    alpha=1.0,
                    size=0.7 * chart_scale,
                    palette=sns.color_palette(stripplot_color_list[:-1]),
                    ax=ax,
                ).set(xlabel=None, ylabel=None)

                # set tick label font size
                ax.tick_params(axis="both", colors=style.style_grey, labelsize=1.2 * chart_scale)

                # save plots or show
                if save_plots:
                    plot_path = os.path.join(
                        self.training_plots_parameter_selection_dir,
                        f"{estimator_class}_{param}.jpg"
                    )
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()
                else:
                    plt.show()

        # otherwise treat it as a numeric parameter
        else:
            # cast "iteration" as an int and the param values as float
            convert_dict = {"iteration": int, param: float}
            actual_iter_df = actual_iter_df.astype(convert_dict)

            # create color map
            color_list = style.color_gen(name=color_map, num=3)

            # create prettierplot object
            p = PrettierPlot(chart_scale=chart_scale, plot_orientation = "wide_narrow")

            # add canvas to prettierplot object
            ax = p.make_canvas(
                title=f"Selection vs. theoretical distribution\n* {estimator_class} - {param}",
                y_shift=0.8,
                position=121,
                title_scale=title_scale,
            )

            # dynamically set x-unit precision based on max value
            if -1.0 <= np.nanmax(theoretical_dist) <= 1.0:
                x_units = "fff"
            elif 1.0 < np.nanmax(theoretical_dist) <= 5.0:
                x_units = "ff"
            elif np.nanmax(theoretical_dist) > 5.0:
                x_units = "f"

            # add kernsel density plot for theoretical distribution to canvas
            p.kde_plot(
                theoretical_dist,
                color=color_list[0],
                y_units="ffff",
                x_units=x_units,
                line_width=0.4,
                bw=0.4,
                ax=ax,
            )

            # add kernsel density plot for actual distribution to canvas
            p.kde_plot(
                actual_dist,
                color=color_list[1],
                y_units="ffff",
                x_units=x_units,
                line_width=0.4,
                bw=0.4,
                ax=ax,
            )

            ## create custom legend
            # create labels
            label_color = {}
            legend_labels = ["Theoretical", "Actual"]
            for ix, i in enumerate(legend_labels):
                label_color[i] = color_list[ix]

            # create legend Patches
            Patches = [Patch(color=v, label=k, alpha=1.0) for k, v in label_color.items()]

            # draw legend
            leg = plt.legend(
                handles=Patches,
                fontsize=1.1 * chart_scale,
                loc="upper right",
                markerscale=0.6 * chart_scale,
                ncol=1,
                bbox_to_anchor=(.95, 1.1),
            )

            # label font color
            for text in leg.get_texts():
                plt.setp(text, color="grey")

            # dynamically set y-unit precision based on max value
            if -1.0 <= np.nanmax(actual_iter_df[param]) <= 1.0:
                y_units = "fff"
            elif 1.0 < np.nanmax(actual_iter_df[param]) <= 5.0:
                y_units = "ff"
            elif np.nanmax(actual_iter_df[param]) > 5.0:
                y_units = "f"

            # add canvas to prettierplot object
            ax = p.make_canvas(
                title=f"Selection by iteration\n* {estimator_class} - {param}",
                y_shift=0.8,
                position=122,
                title_scale=title_scale,
            )

            # add regression plot to canvas
            p.reg_plot(
                x="iteration",
                y=param,
                data=actual_iter_df,
                y_units=y_units,
                x_units="f",
                line_color=color_list[0],
                line_width=0.4,
                dot_color=color_list[1],
                dot_size=10.0,
                alpha=0.6,
                ax=ax
            )

            # save plots or show
            if save_plots:
                plot_path = os.path.join(
                    self.training_plots_parameter_selection_dir,
                    f"{estimator_class}_{param}.jpg"
                )
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
            else:
                plt.show()

def sample_plot(self, sample_space, n_iter, chart_scale=15):
    """
    Documentation:

        ---
        Definition:
            Visualizes a single hyperopt theoretical distribution. Useful for helping to determine a
            distribution to use when setting up hyperopt distribution objects for actual parameter
            tuning.

        ---
        Parameters:
            sample_space : dictionary
                Dictionary of 'param name: hyperopt distribution object' key/value pairs. The name can
                be arbitrarily chosen, and the value is a defined hyperopt distribution.
            n_iter : int
                Number of iterations to draw from theoretical distribution in order to visualize the
                theoretical distribution. Higher number leads to more robust distribution but can take
                considerably longer to create.
            chart_scale : float, default=15
                Controls proportions of visualizations. larger values scale visual up in size, smaller values
                scale visual down in size.

    """
    # iterate through each parameter
    for param in sample_space.keys():

        # sample from theoretical distribution for n_iters
        theoretical_dist = []
        for _ in range(n_iter):
            theoretical_dist.append(sample(sample_space)[param])
        theoretical_dist = np.array(theoretical_dist)

        # create prettierplot object
        p = PrettierPlot(chart_scale=chart_scale)

        # add canvas to prettierplot object
        ax = p.make_canvas(
            title="actual vs. theoretical plot\n* {}".format(param),
            y_shift=0.8,
            position=111,
        )

        # add kernel density plot to canvas
        p.kde_plot(
            theoretical_dist,
            color=style.style_grey,
            y_units="p",
            x_units="fff" if np.nanmax(theoretical_dist) <= 5.0 else "ff",
            ax=ax,
        )

def model_type_check(estimator, n_jobs, params=None):
    """
    Documentation:

        ---
        Description:
            Detects type of estimator and create model object. Also returns
            the name of the estimator.

        ---
        Parameters:
            estimator : str, sklearn api object, or instantiated model
                Estimator to build.
            n_jobs : int, default=1
                Number of workers to deploy upon execution, if applicable. If estimator does not
                have an n_jobs parameter, this is ignored.
    """
    # if estimator is passed as str, eval str to create estimator
    if isinstance(estimator, str):
        estimator = eval(estimator)

    # if estimator is an api object, pass through BasicModelBuilder
    if isinstance(estimator, type) or isinstance(estimator, abc.ABCMeta):
        model = BasicModelBuilder(estimator_class=estimator, n_jobs=n_jobs, params=params)
        estimator_name = model.estimator_name

    # otherwise clone the instantiated model that was passed
    else:
        model = clone(estimator)
        estimator_name = retrieve_variable_name(estimator)

    return model, estimator_name

def retrieve_variable_name(variable):

    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is variable]
        if len(names) > 0:
            return names[0]
