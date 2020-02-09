import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    explained_variance_score,
    mean_squared_log_error,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    roc_curve,
    accuracy_score,
    roc_auc_score,
    homogeneity_score,
    completeness_score,
    classification_report,
    silhouette_samples,
)
from sklearn.model_selection import (
    KFold,
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)



def top_bayes_optim_models(self, bayes_optim_summary, num_models=1):
    """
    documentation:
        description:
            aggregate best model(s) for each estimator as determined by bayesian optimization
            hyperparamter tuning process.
        paramaters:
            bayes_optim_summary : pandas DataFrame
                pandas DataFrame containing results from bayesian optimization process
                execution.
            num_models : int, default=1
                number of top models to return per estimator.
        returns:
            results : dictionary
                dictionary containing string: list key/value pairs, where the string is the
                name of a algorithm class and the list contains the integer(s) associated with
                the best model(s) as identified in the hyperparameter optimization summary.
    """

    models = {}
    for estimator in bayes_optim_summary["estimator"].unique():
        est_df = bayes_optim_summary[
            bayes_optim_summary["estimator"] == estimator
        ].sort_values(
            ["loss", "std_score", "train_time"], ascending=[True, True, True]
        )[
            "iteration"
        ][
            :num_models
        ]
        models[estimator] = est_df.values.tolist()
    return models


def regression_stats(self, model, y_true, y_pred, feature_count, fold=0, data_type="training"):
    """
    documentation:
        description:
            create a dictionary containing information regarding model information and various metrics
            describing the model's performance.
        paramaters:
            model : model object
                instantiated model object.
            y_true : pandas DataFrame or array
                True labels.
            y_pred : Pandas Series or array
                predicted labels.
            feature_count : int
                number of features in the observation data. used to calculate adjusted r_squared.
            fold : int, default=0
                indicator for which cross_validation fold the performance is associated with. if 0,
                it is assumed that the evaluation is on an entire dataset (either the full training
                dataset or the full validation dataset), as opposed to a fold.
        returns:
            results : dictionary
                dictionary containing string: float key/value pairs, where the string is the

    """
    results = {}

    results["estimator"] = model.estimator.__name__
    results["parameter_set"] = model.model_iter
    results["data_type"] = data_type
    results["fold"] = fold
    results["n"] = len(y_true)

    results["explained_variance"] = explained_variance_score(y_true, y_pred)
    results["msle"] = mean_squared_log_error(y_true, y_pred)
    results["mean_ae"] = mean_absolute_error(y_true, y_pred)
    results["median_ae"] = median_absolute_error(y_true, y_pred)
    results["mse"] = mean_squared_error(y_true, y_pred)
    results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    results["r2"] = r2_score(y_true, y_pred)
    results["adjusted_r2"] = 1 - (1 - r2_score(y_true, y_pred)) * (
        len(y_true) - 1
    ) / (len(y_true) - feature_count - 1)
    return results


def regression_results(self, model, X_train, y_train, X_valid=None, y_valid=None, n_folds=3,
                        random_state=1, feature_selector_summary=None):
    """
    documentation:
        description:
            creates a pandas DataFrame where each row captures various summary statistics pertaining to a model's performance.
            captures performance data for training and validation datasets. if no validation set is provided, then
            cross_validation is performed on the training dataset.
        paramaters:
            model : model object
                instantiated model object.
            X_train : pandas DataFrame
                training data observations.
            y_train : Pandas Series
                training data labels.
            X_valid : pandas DataFrame, default=None
                validation data observations.
            y_valid : Pandas Series, default=None
                validation data labels.
            n_folds : int, default=3
                number of cross_validation folds to use when generating
                cv roc graph.
            random_state : int, default=1
                random number seed.
            feature_selector_summary : pndas DataFrame, default=None
                pandas DataFrame containing various summary statistics pertaining to model performance. if none, returns summary
                pandas DataFrame for the input model. if feature_selector_summary DataFrame is provided from a previous run, the new
                performance results are appended to the provivded summary.
        returns:
            feature_selector_summary : pndas DataFrame
                dataframe containing various summary statistics pertaining to model performance.
    """
    model.fit(X_train.values, y_train.values)

    ## training dataset
    y_pred = model.predict(X_train.values)
    results = self.regression_stats(
        model=model,
        y_true=y_train.values,
        y_pred=y_pred,
        feature_count=X_train.shape[1],
    )
    # create shell results DataFrame and append
    if feature_selector_summary is None:
        feature_selector_summary = pd.DataFrame(columns=list(results.keys()))
    feature_selector_summary = feature_selector_summary.append(
        results, ignore_index=True
    )

    ## validation dataset
    # if validation data is provided...
    if X_valid is not None:
        y_pred = model.predict(X_valid.values)
        results = self.regression_stats(
            model=model,
            y_true=y_valid.values,
            y_pred=y_pred,
            feature_count=X_train.shape[1],
            data_type="validation",
        )
        feature_selector_summary = feature_selector_summary.append(
            results, ignore_index=True
        )
    else:
        # if validation data is not provided, then perform k_fold cross validation on
        # training data
        cv = list(
            KFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            ).split(X_train, y_train)
        )

        for i, (train_ix, valid_ix) in enumerate(cv):
            X_train_cv = X_train.iloc[train_ix]
            y_train_cv = y_train.iloc[train_ix]
            X_valid_cv = X_train.iloc[valid_ix]
            y_valid_cv = y_train.iloc[valid_ix]

            y_pred = model.fit(X_train_cv.values, y_train_cv.values).predict(
                X_valid_cv.values
            )
            results = self.regression_stats(
                model=model,
                y_true=y_valid_cv,
                y_pred=y_pred,
                feature_count=X_valid_cv.shape[1],
                data_type="validation",
                fold=i + 1,
            )
            feature_selector_summary = feature_selector_summary.append(
                results, ignore_index=True
            )
    return feature_selector_summary
