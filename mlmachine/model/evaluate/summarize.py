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
    Documentation:
        Description:
            aggregate best model(s) for each estimator as determined by bayesian optimization
            hyperparamter tuning process.
        paramaters:
            bayes_optim_summary : Pandas DataFrame
                Pandas DataFrame containing results from bayesian optimization process
                execution.
            num_models : int, default=1
                number of top models to return per estimator.
        Returns:
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

def binary_prediction_summary(self, model, X_train, y_train, X_valid=None, y_valid=None):
    """
    Documentation:
        Description:
            creates a Pandas DataFrame where each row corresponds to an observation, the model's prediction
            for that observation, and the probabilities associated with the prediction.
        paramaters:
            model : model object
                instantiated model object.
            X_train : Pandas DataFrame
                training data observations.
            y_train : Pandas Series
                training data labels.
            X_valid : Pandas DataFrame, default=None
                validation data observations.
            y_valid : Pandas Series, default=None
                validation data labels.
        Returns:
            df : Pandas DataFrame
                Pandas DataFrame containing prediction summary data.
    """
    model.fit(X_train.values, y_train.values)

    if X_valid is None:
        probas = model.predict_proba(X_train)
        data = {
            "Label": y_train,
            "Prediction": probas.argmax(axis=1),
            "Positive": probas[:,0],
            "Negative": probas[:,1],
        }
        df = pd.DataFrame(data, index=y_train.index)
        df["Difference"] = np.abs(df["Positive"] - df["Negative"])
        df["Incorrect"] = np.abs(df["Label"] - df["Prediction"])
    else:
        probas = model.predict_proba(X_valid)
        data = {
            "Label": y_valid,
            "Prediction": probas.argmax(axis=1),
            "Positive": probas[:,0],
            "Negative": probas[:,1],
        }
        df = pd.DataFrame(data, index=y_valid.index)
        df["Difference"] = np.abs(df["Positive"] - df["Negative"])
        df["Incorrect"] = np.abs(df["Label"] - df["Prediction"])

    # sort to bring largest errors to the top
    df = df.sort_values(["Incorrect","Difference"], ascending=[False,False])
    return df

def regression_prediction_summary(self, model, X_train, y_train, X_valid=None, y_valid=None):
    """
    Documentation:
        Description:
            creates a Pandas DataFrame where each row corresponds to an observation, the model's prediction
            for that observation, and the probabilities associated with the prediction.
        paramaters:
            model : model object
                instantiated model object.
            X_train : Pandas DataFrame
                training data observations.
            y_train : Pandas Series
                training data labels.
            X_valid : Pandas DataFrame, default=None
                validation data observations.
            y_valid : Pandas Series, default=None
                validation data labels.
        Returns:
            df : Pandas DataFrame
                Pandas DataFrame containing prediction summary data.
    """
    if X_valid is None:
        preds = model.predict(X_train)
        data = {
            "Label": y_train,
            "Prediction": preds,
        }
        df = pd.DataFrame(data, index=y_train.index)
        df["Difference"] = np.abs(df["Label"] - df["Prediction"])
        df["Percent difference"] =  ((df['Prediction'] - df['Label']) / df['Label']) * 100
    else:
        preds = model.predict(X_valid)
        data = {
            "Label": y_valid,
            "Prediction": preds,
        }
        df = pd.DataFrame(data, index=y_valid.index)
        df["Difference"] = np.abs(df["Label"] - df["Prediction"])
        df["Percent difference"] =  ((df['Prediction'] - df['Label']) / df['Label']) * 100

    # sort to bring largest errors to the top
    df = df.sort_values(["Percent difference"], ascending=[False])
    return df

def regression_stats(self, model, y_true, y_pred, feature_count, fold=0, data_type="training"):
    """
    Documentation:
        Description:
            create a dictionary containing information regarding model information and various metrics
            describing the model's performance.
        paramaters:
            model : model object
                instantiated model object.
            y_true : Pandas DataFrame or array
                True labels.
            y_pred : Pandas Series or array
                predicted labels.
            feature_count : int
                number of features in the observation data. used to calculate adjusted r_squared.
            fold : int, default=0
                indicator for which cross_validation fold the performance is associated with. if 0,
                it is assumed that the evaluation is on an entire dataset (either the full training
                dataset or the full validation dataset), as opposed to a fold.
        Returns:
            results : dictionary
                dictionary containing string: float key/value pairs, where the string is the

    """
    results = {}

    results["Estimator"] = model.estimator_name
    results["Parameter set"] = model.model_iter
    results["Dataset type"] = data_type
    results["CV fold"] = fold
    results["N"] = len(y_true)

    results["Explained Variance"] = explained_variance_score(y_true, y_pred)
    results["Mean squared log error"] = mean_squared_log_error(y_true, y_pred)
    results["Mean absolute error"] = mean_absolute_error(y_true, y_pred)
    results["Median absolute error"] = median_absolute_error(y_true, y_pred)
    results["Mean squared error"] = mean_squared_error(y_true, y_pred)
    results["Root mean squared error"] = np.sqrt(mean_squared_error(y_true, y_pred))
    results["R-squared"] = r2_score(y_true, y_pred)
    results["Adjusted R-squared"] = 1 - (1 - r2_score(y_true, y_pred)) * (
        len(y_true) - 1
    ) / (len(y_true) - feature_count - 1)
    return results

def regression_results(self, model, X_train, y_train, X_valid=None, y_valid=None, n_folds=None,
                        random_state=1, regression_results_summary=None):
    """
    Documentation:
        Description:
            creates a Pandas DataFrame where each row captures various summary statistics pertaining to a model's performance.
            captures performance data for training and validation datasets. if no validation set is provided, then
            cross_validation is performed on the training dataset.
        paramaters:
            model : model object
                instantiated model object.
            X_train : Pandas DataFrame
                training data observations.
            y_train : Pandas Series
                training data labels.
            X_valid : Pandas DataFrame, default=None
                validation data observations.
            y_valid : Pandas Series, default=None
                validation data labels.
            n_folds : int, default=None
                number of cross_validation folds to use when generating
                cv roc graph.
            random_state : int, default=1
                random number seed.
            regression_results_summary : Pandas DataFrame, default=None
                Pandas DataFrame containing various summary statistics pertaining to model performance. if none, returns summary
                Pandas DataFrame for the input model. if regression_results_summary DataFrame is provided from a previous run, the new
                performance results are appended to the provivded summary.
        Returns:
            regression_results_summary : Pandas DataFrame
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
    if regression_results_summary is None:
        regression_results_summary = pd.DataFrame(columns=list(results.keys()))
    regression_results_summary = regression_results_summary.append(
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
        regression_results_summary = regression_results_summary.append(
            results, ignore_index=True
        )
    elif isinstance(n_folds, int):
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
            regression_results_summary = regression_results_summary.append(
                results, ignore_index=True
            )
            
    return regression_results_summary
