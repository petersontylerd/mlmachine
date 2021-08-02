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


def top_bayes_optim_models(self, bayes_optim_summary, metric, num_models=1):
    """
    Documentation:

        ---
        Description:
            Aggregate best model(s) for each estimator as determined by bayesian optimization
            hyperparamter tuning process.

        ---
        Paramaters:
            bayes_optim_summary : Pandas DataFrame
                Pandas DataFrame containing results from bayesian optimization process.
            metric : str
                Column to use when determining the top performing model.
            num_models : int, default=1
                Number of top models to return per estimator.

        ---
        Returns:
            models : dictionary
                Dictionary containing string: list key/value pairs, where the string is the
                name of an estimator and the list contains the integer(s) associated with
                the best model(s) as identified in the hyperparameter optimization summary.
    """

    models = {}

    if metric in ["validation_score"]:
        sort_direction = False
    elif metric in ["loss"]:
        sort_direction = True


    # iterate through unique estimators in bayes_optim_summary
    for estimator in bayes_optim_summary["estimator"].unique():

        # subset bayes_optim_summary by current estimator and keep N rows, where
        # N = num_models. sort by loss and capture only "iteration" values
        est_df = bayes_optim_summary[
            bayes_optim_summary["estimator"] == estimator
        ].sort_values(
            [metric, "std_score", "train_time"], ascending=[sort_direction, True, True]
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

        ---
        Description:
            Creates a Pandas DataFrame where each row corresponds to an observation, the model's prediction
            for that observation, the true label, and the probabilities associated with the prediction.

        ---
        Paramaters:
            model : model object
                Instantiated model object.
            X_train : Pandas DataFrame
                Training data observations.
            y_train : Pandas Series
                Training target data.
            X_valid : Pandas DataFrame, default=None
                Validation data observations.
            y_valid : Pandas Series, default=None
                Validation target data.

        ---
        Returns:
            df : Pandas DataFrame
                Pandas DataFrame containing prediction summary data.
    """
    # fit model on training data
    model.fit(X_train.values, y_train.values)

    # if no validation data is provided
    if X_valid is None:

        # capture probabilites based on training data
        probas = model.predict_proba(X_train)

        # organize probabilities in dictionary
        data = {
            "Label": y_train,
            "Prediction": probas.argmax(axis=1),
            "Positive": probas[:,0],
            "Negative": probas[:,1],
        }

        # capture data in a DataFrame
        df = pd.DataFrame(data, index=y_train.index)

        # add "Difference" and "Incorrect" summary columns
        df["Difference"] = np.abs(df["Positive"] - df["Negative"])
        df["Incorrect"] = np.abs(df["Label"] - df["Prediction"])

    else:
        # capture probabilites based on validation data
        probas = model.predict_proba(X_valid)

        # organize probabilities in dictionary
        data = {
            "Label": y_valid,
            "Prediction": probas.argmax(axis=1),
            "Positive": probas[:,0],
            "Negative": probas[:,1],
        }

        # capture data in a DataFrame
        df = pd.DataFrame(data, index=y_train.index)

        # add "Difference" and "Incorrect" summary columns
        df["Difference"] = np.abs(df["Positive"] - df["Negative"])
        df["Incorrect"] = np.abs(df["Label"] - df["Prediction"])

    # sort to bring largest errors to the top
    df = df.sort_values(["Incorrect","Difference"], ascending=[False,False])
    return df

def regression_prediction_summary(self, model, X_train, y_train, X_valid=None, y_valid=None):
    """
    Documentation:

        ---
        Description:
            Creates a Pandas DataFrame where each row corresponds to an observation, the model's prediction
            for that observation, and the true label.

        ---
        Paramaters:
            model : model object
                Instantiated model object.
            X_train : Pandas DataFrame
                Training target data.
            y_train : Pandas Series
                Training data labels.
            X_valid : Pandas DataFrame, default=None
                Validation data observations.
            y_valid : Pandas Series, default=None
                Validation target data.

        ---
        Returns:
            df : Pandas DataFrame
                Pandas DataFrame containing prediction summary data.
    """
    # fit model on training data
    model.fit(X_train.values, y_train.values)

    # if no validation data is provided
    if X_valid is None:

        # generate predictions using training data
        preds = model.predict(X_train)

        # organize data in a dictionary
        data = {
            "Label": y_train,
            "Prediction": preds,
        }

        # capture data in a DataFrame
        df = pd.DataFrame(data, index=y_train.index)

        # add "Difference" and "Incorrect" summary columns
        df["Difference"] = np.abs(df["Label"] - df["Prediction"])
        df["Percent difference"] =  ((df['Prediction'] - df['Label']) / df['Label']) * 100

    else:
        # generate predictions using validation data
        preds = model.predict(X_valid)

        # organize data in a dictionary
        data = {
            "Label": y_valid,
            "Prediction": preds,
        }

        # capture data in a DataFrame
        df = pd.DataFrame(data, index=y_train.index)

        # add "Difference" and "Incorrect" summary columns
        df["Difference"] = np.abs(df["Label"] - df["Prediction"])
        df["Percent difference"] =  ((df['Prediction'] - df['Label']) / df['Label']) * 100

    # sort to bring largest errors to the top
    df = df.sort_values(["Percent difference"], ascending=[False])
    return df

def regression_stats(self, model, y_true, y_pred, feature_count, fold=0, data_type="training"):
    """
    Documentation:

        ---
        Description:
            Create a dictionary containing information regarding model information and various metrics
            describing the model's performance.

        ---
        Paramaters:
            model : model object
                Instantiated model object.
            y_true : Pandas Series or array
                True labels.
            y_pred : Pandas Series or array
                Predicted labels.
            feature_count : int
                Number of features in the observation data. Used to calculate adjusted R-squared.
            fold : int, default=0
                Indicator for which cross-validation fold the performance is associated with. If 0,
                it is assumed that the evaluation is on an entire dataset (either the full training
                dataset or the full validation dataset), as opposed to a fold.
            data_type : str, default="training"
                String describing type of dataset

        ---
        Returns:
            results : dictionary
                Dictionary containing string: float key/value pairs, where the string is the metric name
                and the value is the metric score

    """
    results = {}

    # model information
    results["Estimator"] = model.estimator_name
    results["Parameter set"] = model.model_iter
    results["Dataset type"] = data_type
    results["CV fold"] = fold
    results["N"] = len(y_true)

    # model performance
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

        ---
        Description:
            Creates a Pandas DataFrame where each row captures various summary statistics pertaining to a
            model's performance. Captures performance data for training and validation datasets. If no
            validation set is provided, then cross-validation is performed on the training dataset.

        ---
        Paramaters:
            model : model object
                Instantiated model object.
            X_train : Pandas DataFrame
                Training data observations.
            y_train : Pandas Series
                Training target data.
            X_valid : Pandas DataFrame, default=None
                Validation data observations.
            y_valid : Pandas Series, default=None
                Validation target data.
            n_folds : int, default=None
                Number of cross-validation folds to use when generating
                cv roc graph.
            random_state : int, default=1
                Random number seed.
            regression_results_summary : Pandas DataFrame, default=None
                Pandas DataFrame containing various summary statistics pertaining to model performance. If
                None, returns summary Pandas DataFrame for the input model. If regression_results_summary
                DataFrame is provided from a previous run, the new performance results are appended to the
                provivded summary.

        ---
        Returns:
            regression_results_summary : Pandas DataFrame
                Dataframe containing various summary statistics pertaining to model performance.
    """
    # fit model on training data
    model.fit(X_train.values, y_train.values)

    ## training dataset
    # generate predictions using training data
    y_pred = model.predict(X_train.values)

    # return regression_stats results for training data and predictions
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

        # generate predictions using validation data
        y_pred = model.predict(X_train.values)

        # return regression_stats results for validation data and predictions
        results = self.regression_stats(
            model=model,
            y_true=y_train.values,
            y_pred=y_pred,
            feature_count=X_train.shape[1],
            data_type="validation",
        )

        # append results to DataFrame
        regression_results_summary = regression_results_summary.append(
            results, ignore_index=True
        )
    elif isinstance(n_folds, int):
        # if validation data is not provided, then perform KFold cross-validation on
        # training data
        cv = list(
            KFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            ).split(X_train, y_train)
        )

        # iterate through KFolds
        for i, (train_ix, valid_ix) in enumerate(cv):
            X_train_cv = X_train.iloc[train_ix]
            y_train_cv = y_train.iloc[train_ix]
            X_valid_cv = X_train.iloc[valid_ix]
            y_valid_cv = y_train.iloc[valid_ix]

            # fit model on training observations and make predictions with holdout observations
            y_pred = model.fit(X_train_cv.values, y_train_cv.values).predict(
                X_valid_cv.values
            )

            # return regression_stats results for holdout data and predictions
            results = self.regression_stats(
                model=model,
                y_true=y_valid_cv,
                y_pred=y_pred,
                feature_count=X_valid_cv.shape[1],
                data_type="validation",
                fold=i + 1,
            )

            # append results to DataFrame
            regression_results_summary = regression_results_summary.append(
                results, ignore_index=True
            )

    return regression_results_summary
