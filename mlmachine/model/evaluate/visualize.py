import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import (
    KFold,
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
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
    plot_confusion_matrix,
)

from scipy import stats

from prettierplot.plotter import PrettierPlot
from prettierplot import style


def binary_classification_panel(self, model, X_train, y_train, X_valid=None, y_valid=None, labels=None,
                        n_folds=None, title_scale=1.0, color_map="viridis", random_state=1, chart_scale=15):
    """
    Documentation:

        ---
        Description:
            Generate a panel of reports and visualizations summarizing the
            performance of a classification model.

        ---
        Parameters:
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
            labels : list, default=None
                Custom labels for confusion matrix axes. If left as none,
                will default to 0, 1, 2...
            n_folds : int, default=None
                Number of cross-validation folds to use. If validation data is provided through
                X_valid/y_valid, n_folds is ignored.
            color_map : string specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            title_scale : float, default=1.0
                Controls the scaling up (higher value) and scaling down (lower value) of the size
                of the main chart title, the x_axis title and the y_axis title.
            random_state : int, default=1
                random number seed.
            chart_scale : int or float, default=15
                Controls size and proportions of chart and chart elements. Higher value creates
                larger plots and increases visual elements proportionally.
    """
    print("*" * 55)
    print("* Estimator: {}".format(model.estimator_name))
    print("* Parameter set: {}".format(model.model_iter))
    print("*" * 55)

    print("\n" + "*" * 55)
    print("Training data evaluation\n")

    ## training panel
    # fit model on training data and generate predictions using training data
    y_pred = model.fit(X_train, y_train).predict(X_train)

    # print and generate classification_report using training data
    print(
            classification_report(
                y_train,
                y_pred,
                target_names=labels if labels is not None else np.unique(y_train.values),
            )
        )

    # create prettierplot object
    p = PrettierPlot(chart_scale=chart_scale, plot_orientation="wide_narrow")

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title="Confusion matrix - training data\nModel: {}\nParameter set: {}".format(
            model.estimator_name, model.model_iter
        ),
        y_shift=0.4,
        x_shift=0.25,
        position=121,
        title_scale=title_scale,
    )

    # add confusion plot to canvas
    plot_confusion_matrix(
        estimator=model,
        X=X_train,
        y_true=y_train,
        display_labels=labels if labels is not None else np.unique(y_train.values),
        cmap=color_map,
        values_format=".0f",
        ax=ax,
    )

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title="ROC curve - training data\nModel: {}\nParameter set: {}".format(
            model.estimator_name,
            model.model_iter,
        ),
        x_label="False positive rate",
        y_label="True positive rate",
        y_shift=0.35,
        position=122,
        title_scale=title_scale,
    )
    # add ROC curve to canvas
    p.roc_curve_plot(
        model=model,
        X_train=X_train,
        y_train=y_train,
        linecolor=style.style_grey,
        ax=ax,
    )
    plt.subplots_adjust(wspace=0.3)
    plt.show()

    # if validation data is provided
    if X_valid is not None:
        print("\n" + "*" * 55)
        print("Validation data evaluation\n")

        # fit model on training data and generate predictions using validation data
        y_pred = model.fit(X_train, y_train).predict(X_valid)

        # print and generate classification_report using training data
        print(
            classification_report(
                y_valid,
                y_pred,
                target_names=labels if labels is not None else np.unique(y_train.values),
            )
        )

        # create prettierplot object
        p = PrettierPlot(chart_scale=chart_scale, plot_orientation="wide_narrow")

        # add canvas to prettierplot object
        ax = p.make_canvas(
            title="Confusion matrix - validation data\nModel: {}\nParameter set: {}".format(
                model.estimator_name, model.model_iter
            ),
            y_shift=0.4,
            x_shift=0.25,
            position=121,
            title_scale=title_scale,
        )

        # add confusion matrix to canvas
        plot_confusion_matrix(
            estimator=model,
            X=X_valid,
            y_true=y_valid,
            display_labels=labels if labels is not None else np.unique(y_train.values),
            cmap=color_map,
            values_format=".0f",
            ax=ax,
        )

        # add canvas to prettierplot object
        ax = p.make_canvas(
            title="ROC curve - validation data\nModel: {}\nParameter set: {}".format(
                model.estimator_name,
                model.model_iter,
            ),
            x_label="False positive rate",
            y_label="True positive rate",
            y_shift=0.35,
            position=122,
            # position=111 if X_valid is not None else 121,
            title_scale=title_scale,
        )
        # add ROC curve to canvas
        p.roc_curve_plot(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            linecolor=style.style_grey,
            ax=ax,
        )
        plt.subplots_adjust(wspace=0.3)
        plt.show()

    # if n_folds are provided, indicating cross-validation
    elif isinstance(n_folds, int):
        print("\n" + "*" * 55)
        print("Cross validation evaluation\n")

        # generate cross-validation indices
        cv = list(
            StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            ).split(X_train, y_train)
        )

        # generate colors
        color_list = style.color_gen(color_map, num=len(cv))

        # iterate through cross-validation indices
        for i, (train_ix, valid_ix) in enumerate(cv):
            print("\n" + "*" * 55)
            print("CV Fold {}\n".format(i + 1))

            X_train_cv = X_train.iloc[train_ix]
            y_train_cv = y_train.iloc[train_ix]
            X_valid_cv = X_train.iloc[valid_ix]
            y_valid_cv = y_train.iloc[valid_ix]

            # fit model on training data and generate predictions using holdout observations
            y_pred = model.fit(X_train_cv, y_train_cv).predict(X_valid_cv)

            # print and generate classification_report using holdout observations
            print(
            classification_report(
                    y_valid_cv,
                    y_pred,
                    target_names=labels if labels is not None else np.unique(y_train.values),
                )
            )

            # create prettierplot object
            p = PrettierPlot(chart_scale=chart_scale, plot_orientation="wide_narrow")

            # add canvas to prettierplot object
            ax = p.make_canvas(
                title="Confusion matrix - CV Fold {}\nModel: {}\nParameter set: {}".format(
                    i + 1, model.estimator_name, model.model_iter
                ),
                y_shift=0.4,
                x_shift=0.25,
                position=121,
                title_scale=title_scale,
            )

            # add confusion matrix to canvas
            plot_confusion_matrix(
                estimator=model,
                X=X_valid_cv,
                y_true=y_valid_cv,
                display_labels=labels if labels is not None else np.unique(y_train.values),
                cmap=color_map,
                values_format=".0f",
                ax=ax,
            )

            # add canvas to prettierplot object
            ax = p.make_canvas(
                title="ROC curve - CV Fold {}\nModel: {}\nParameter set: {}".format(
                    i + 1,
                    model.estimator_name,
                    model.model_iter,
                ),
                x_label="False positive rate",
                y_label="True positive rate",
                y_shift=0.35,
                position=122,
                title_scale=title_scale,
            )
            
            # add ROC curve to canvas
            p.roc_curve_plot(
                model=model,
                X_train=X_train_cv,
                y_train=y_train_cv,
                X_valid=X_valid_cv,
                y_valid=y_valid_cv,
                linecolor=style.style_grey,
                ax=ax,
            )
            plt.subplots_adjust(wspace=0.3)
            plt.show()

def regression_panel(self, model, X_train, y_train, X_valid=None, y_valid=None, n_folds=None, title_scale=1.0,
                    color_map="viridis", random_state=1, chart_scale=15):
    """
    Documentation:
        Description:
            creates a set of residual plots and pandas DataFrames, where each row captures various summary statistics
            pertaining to a model's performance. generates residual plots and captures performance data for training
            and validation datasets. If no validation set is provided, then cross_validation is performed on the
            training dataset.
        Parameters:
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
                Number of cross-validation folds to use. If validation data is provided through
                X_valid/y_valid, n_folds is ignored.
            title_scale : float, default=1.0
                Controls the scaling up (higher value) and scaling down (lower value) of the size of
                the main chart title, the x_axis title and the y_axis title.
            color_map : string specifying built_in matplotlib colormap, default="viridis"
                Color map applied to plots.
            random_state : int, default=1
                random number seed.
            chart_scale : int or float, default=15
                Controls size and proportions of chart and chart elements. Higher value creates larger plots
                and increases visual elements proportionally.
    """

    print("*" * 55)
    print("* Estimator: {}".format(model.estimator_name))
    print("* Parameter set: {}".format(model.model_iter))
    print("*" * 55)

    print("\n" + "*" * 55)
    print("Training data evaluation")

    # fit model on training data
    model.fit(X_train.values, y_train.values)

    ## training dataset
    # generate predictions using training data and calculate residuals
    y_pred = model.predict(X_train.values)
    residuals = y_pred - y_train.values

    # create prettierplot object
    p = PrettierPlot(plot_orientation="wide_narrow")
    
    # add canvas to prettierplot object
    ax = p.make_canvas(
        title="Residual plot - training data\nModel: {}\nParameter set: {}".format(
            model.estimator_name,
            model.model_iter,
        ),
        x_label="Predicted values",
        y_label="Residuals",
        y_shift=0.55,
        title_scale=title_scale,
        position=121,
    )

    # dynamically size precision of x-units based on magnitude of maximum
    # predicted values
    if -1 <= np.nanmax(y_pred) <= 1:
        x_units = "fff"
    elif -100 <= np.nanmax(y_pred) <= 100:
        x_units = "ff"
    else:
        x_units = "f"

    # dynamically size precision of y-units based on magnitude of maximum
    # predicted values
    if -0.1 <= np.nanmax(residuals) <= 0.1:
        y_units = "ffff"
    elif -1 <= np.nanmax(residuals) <= 1:
        y_units = "fff"
    elif -10 <= np.nanmax(residuals) <= 10:
        y_units = "ff"
    else:
        y_units = "f"

    # x tick label rotation
    if -10000 < np.nanmax(y_pred) < 10000:
        x_rotate = 0
    else:
        x_rotate = 45

    # add 2-dimensional scatter plot to canvas
    p.scatter_2d(
        x=y_pred,
        y=residuals,
        size=7,
        color=style.style_grey,
        y_units=y_units,
        x_units=x_units,
        ax=ax,
    )

    # plot horizontal line at y=0
    plt.hlines(
        y=0, xmin=np.min(y_pred), xmax=np.max(y_pred), color=style.style_grey, lw=2
    )

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title="Residual distribution - training data\nModel: {}\nParameter set: {}".format(
            model.estimator_name,
            model.model_iter,
        ),
        title_scale=title_scale,
        position=122,
    )

    # add distribution plot to canvas
    p.dist_plot(
        residuals,
        fit=stats.norm,
        color=style.style_grey,
        y_units="ff",
        x_units="fff",
        ax=ax,
    )
    plt.show()

    # generate regression_stats using training data and predictions
    results = self.regression_stats(
        model=model,
        y_true=y_train.values,
        y_pred=y_pred,
        feature_count=X_train.shape[1],
    )
    
    # create shell results DataFrame and append
    regression_results_summary = pd.DataFrame(columns=list(results.keys()))
    regression_results_summary = regression_results_summary.append(
        results, ignore_index=True
    )

    ## validation dataset
    # if validation data is provided...
    if X_valid is not None:
        print("\n" + "*" * 55)
        print("Training data evaluation")

        # generate predictions with validation data and calculate residuals
        y_pred = model.predict(X_valid.values)
        residuals = y_pred - y_valid.values

        # create prettierplot object
        p = PrettierPlot(plot_orientation="wide_narrow")
        
        # add canvas to prettierplot object
        ax = p.make_canvas(
            title="Residual plot - training data\nModel: {}\nParameter set: {}".format(
                model.estimator_name,
                model.model_iter,
            ),
            x_label="Predicted values",
            y_label="Residuals",
            y_shift=0.55,
            title_scale=title_scale,
            position=121,
        )

        # add 2-dimensional scatter plot to canvas
        p.scatter_2d(
            x=y_pred,
            y=residuals,
            size=7,
            color=style.style_grey,
            y_units=y_units,
            x_units=x_units,
            ax=ax,
        )

        # plot horizontal line at y=0
        plt.hlines(
            y=0, xmin=np.min(y_pred), xmax=np.max(y_pred), color=style.style_grey, lw=2
        )

        # add canvas to prettierplot object
        ax = p.make_canvas(
            title="Residual distribution - training data\nModel: {}\nParameter set: {}".format(
                model.estimator_name,
                model.model_iter,
            ),
            title_scale=title_scale,
            position=122,
        )

        # add distribution plot to canvas
        p.dist_plot(
            residuals,
            fit=stats.norm,
            color=style.style_grey,
            y_units="ff",
            x_units="fff",
            ax=ax,
        )
        plt.show()

        # generate regression_stats using validation data and predictions
        results = self.regression_stats(
            model=model,
            y_true=y_valid.values,
            y_pred=y_pred,
            feature_count=X_train.shape[1],
            data_type="validation",
        )
        
        # append results to regression_results_summary
        regression_results_summary = regression_results_summary.append(
            results, ignore_index=True
        )
        display(regression_results_summary)

    # if n_folds are provided, indicating cross-validation
    elif isinstance(n_folds, int):
        
        # generate cross-validation indices
        cv = list(
            KFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            ).split(X_train, y_train)
        )

        print("\n" + "*" * 55)
        print("Cross validation evaluation")

        # iterate through cross-validation indices
        for i, (train_ix, valid_ix) in enumerate(cv):
            X_train_cv = X_train.iloc[train_ix]
            y_train_cv = y_train.iloc[train_ix]
            X_valid_cv = X_train.iloc[valid_ix]
            y_valid_cv = y_train.iloc[valid_ix]

            # fit model on training data and generate predictions using holdout observations
            y_pred = model.fit(X_train_cv.values, y_train_cv.values).predict(
                X_valid_cv.values
            )
            
            # calculate residuals
            residuals = y_pred - y_valid_cv.values

            # create prettierplot object
            p = PrettierPlot(plot_orientation="wide_narrow")
            
            # add canvas to prettierplot object
            ax = p.make_canvas(
                title="Residual plot - CV fold {}\nModel: {}\nParameter set: {}".format(
                    i + 1,
                    model.estimator_name,
                    model.model_iter,
                ),
                x_label="Predicted values",
                y_label="Residuals",
                y_shift=0.55,
                position=121,
                title_scale=title_scale,
            )

            # add 2-dimensional scatter plot to canvas
            p.scatter_2d(
                x=y_pred,
                y=residuals,
                size=7,
                color=style.style_grey,
                # color=color_list[i],
                y_units=y_units,
                x_units=x_units,
                ax=ax,
            )
            
            # plot horizontal line at y=0
            plt.hlines(
                y=0,
                xmin=np.min(y_pred),
                xmax=np.max(y_pred),
                color=style.style_grey,
                lw=2,
            )

            # add canvas to prettierplot object
            ax = p.make_canvas(
                title="Residual distribution - CV fold {}\nModel: {}\nParameter set: {}".format(
                    i + 1,
                    model.estimator_name,
                    model.model_iter,
                ),
                title_scale=title_scale,
                position=122,
            )

            # add distribution plot to canvas
            p.dist_plot(
                residuals,
                fit=stats.norm,
                color=style.style_grey,
                y_units="ff",
                x_units="fff",
                ax=ax,
            )
            plt.show()

            # generate regression_stats using holdout observations and predictions
            results = self.regression_stats(
                model=model,
                y_true=y_valid_cv,
                y_pred=y_pred,
                feature_count=X_valid_cv.shape[1],
                data_type="validation",
                fold=i + 1,
            )
            
            # append results to regression_results_summary
            regression_results_summary = regression_results_summary.append(
                results, ignore_index=True
            )
        print("\n" + "*" * 55)
        print("Summary")

        display(regression_results_summary)
    else:
        display(regression_results_summary)
