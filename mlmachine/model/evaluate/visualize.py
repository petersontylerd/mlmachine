import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn.model_selection as model_selection
import sklearn.metrics as metrics

from prettierplot.plotter import PrettierPlot
from prettierplot import style

import shap


def classification_panel(
    self,
    model,
    x_train,
    y_train,
    x_valid=None,
    y_valid=None,
    cm_labels=None,
    n_folds=3,
    title_scale=0.7,
    color_map="viridis",
    random_state=1,
):
    """
    documentation:
        description:
            generate a panel of reports and visualizations summarizing the
            performance of a classification model.
        paramaters:
            model : model object
                instantiated model object.
            x_train : pandas DataFrame
                training data observations.
            y_train : pandas series
                training data labels.
            x_valid : pandas DataFrame, default =None
                validation data observations.
            y_valid : pandas series, default =None
                validation data labels.
            cm_labels : list, default =None
                custom labels for confusion matrix axes. if left as none,
                will default to 0, 1, 2...
            n_folds : int, default = 3
                number of cross_validation folds to use when generating
                cv roc graph.
            color_map : string specifying built_in matplotlib colormap, default = "viridis"
                colormap from which to draw plot colors.
            title_scale : float, default = 1.0
                controls the scaling up (higher value) and scaling down (lower value) of the size of
                the main chart title, the x_axis title and the y_axis title.
            random_state : int, default = 1
                random number seed.
    """

    print("*" * 55)
    print("* estimator: {}".format(model.estimator.__name__))
    print("* parameter set: {}".format(model.model_iter))
    print("*" * 55)

    # visualize results with confusion matrix
    p = PrettierPlot(chart_prop=18)
    ax = p.make_canvas(
        title="model: {}\n_parameter set: {}".format(
            model.estimator.__name__, model.model_iter
        ),
        x_label="predicted",
        y_label="actual",
        y_shift=0.4,
        x_shift=0.25,
        position=211,
        title_scale=title_scale,
    )

    # conditional control for which data is used to generate predictions
    if x_valid is not None:
        y_pred = model.fit(x_train, y_train).predict(x_valid)
        print(
            metrics.classification_report(
                y_valid, y_pred, labels=np.unique(y_train.values)
            )
        )
        p.pretty_confusion_matrix(
            y_True=y_valid,
            y_pred=y_pred,
            labels=cm_labels if cm_labels is not None else np.unique(y_train.values),
            ax=None,
        )
    else:
        y_pred = model.fit(x_train, y_train).predict(x_train)
        print(
            metrics.classification_report(
                y_train, y_pred, labels=np.unique(y_train.values)
            )
        )
        p.pretty_confusion_matrix(
            y_True=y_train,
            y_pred=y_pred,
            labels=cm_labels if cm_labels is not None else np.unique(y_train.values),
            ax=None,
        )
    plt.show()

    # standard roc curve for full training dataset or validation dataset
    # if x_valid is passed in as none, generate roc curve for training data
    p = PrettierPlot(chart_prop=15)
    ax = p.make_canvas(
        title="roc curve - {} data\n_model: {}\n_parameter set: {}".format(
            "training" if x_valid is None else "validation",
            model.estimator.__name__,
            model.model_iter,
        ),
        x_label="False positive rate",
        y_label="True positive rate",
        y_shift=0.5,
        position=111 if x_valid is not None else 121,
        title_scale=title_scale,
    )
    p.pretty_roc_curve(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        linecolor=style.style_grey,
        ax=ax,
    )

    # if no validation data is passed, then perform k_fold cross validation and generate
    # an roc curve for each held out validation set.
    if x_valid is None:
        # cross_validated roc curve
        cv = list(
            model_selection.StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            ).split(x_train, y_train)
        )

        # generate colors
        color_list = style.color_gen(color_map, num=len(cv))

        # plot roc curves
        ax = p.make_canvas(
            title="roc curve - validation data, {}_fold cv\n_model: {}\n_parameter set: {}".format(
                n_folds, model.estimator.__name__, model.model_iter
            ),
            x_label="False positive rate",
            y_label="True positive rate",
            y_shift=0.5,
            title_scale=title_scale,
            position=122,
        )
        for i, (train_ix, valid_ix) in enumerate(cv):
            x_train_cv = x_train.iloc[train_ix]
            y_train_cv = y_train.iloc[train_ix]
            x_valid_cv = x_train.iloc[valid_ix]
            y_valid_cv = y_train.iloc[valid_ix]

            p.pretty_roc_curve(
                model=model,
                x_train=x_train_cv,
                y_train=y_train_cv,
                x_valid=x_valid_cv,
                y_valid=y_valid_cv,
                linecolor=color_list[i],
                ax=ax,
            )
    plt.subplots_adjust(wspace=0.3)
    plt.show()


def regression_panel(
    self,
    model,
    x_train,
    y_train,
    x_valid=None,
    y_valid=None,
    n_folds=3,
    title_scale=0.7,
    color_map="viridis",
    random_state=1,
):
    """
    documentation:
        description:
            creates a set of residual plots and pandas DataFrames, where each row captures various summary statistics
            pertaining to a model's performance. generates residual plots and captures performance data for training
            and validation datasets. if no validation set is provided, then cross_validation is performed on the
            training dataset.
        paramaters:
            model : model object
                instantiated model object.
            x_train : pandas DataFrame
                training data observations.
            y_train : pandas series
                training data labels.
            x_valid : pandas DataFrame, default =None
                validation data observations.
            y_valid : pandas series, default =None
                validation data labels.
            n_folds : int, default = 3
                number of cross_validation folds to use when generating
                cv roc graph.
            random_state : int, default = 1
                random number seed.
            title_scale : float, default = 1.0
                controls the scaling up (higher value) and scaling down (lower value) of the size of
                the main chart title, the x_axis title and the y_axis title.
            color_map : string specifying built_in matplotlib colormap, default = "viridis"
                colormap from which to draw plot colors.
    """

    print("*" * 55)
    print("* estimator: {}".format(model.estimator.__name__))
    print("* parameter set: {}".format(model.model_iter))
    print("*" * 55 + "\n")

    model.fit(x_train.values, y_train.values)

    print("*" * 27)
    print("full training dataset performance\n")

    ## training dataset
    y_pred = model.predict(x_train.values)

    # residual plot
    p = PrettierPlot()
    ax = p.make_canvas(
        title="model: {}\n_parameter set: {}\n_training data".format(
            model.estimator.__name__, model.model_iter
        ),
        x_label="predicted values",
        y_label="residuals",
        y_shift=0.7,
        title_scale=title_scale,
    )

    p.pretty2d_scatter(
        x=y_pred, y=y_pred - y_train.values, size=7, color=style.style_grey, ax=ax,
    )
    plt.hlines(
        y=0, xmin=np.min(y_pred), xmax=np.max(y_pred), color=style.style_grey, lw=2
    )
    plt.show()

    # training data results summary
    results = self.regression_stats(
        model=model,
        y_True=y_train.values,
        y_pred=y_pred,
        feature_count=x_train.shape[1],
    )
    # create shell results DataFrame and append
    feature_selector_summary = pd.DataFrame(columns=list(results.keys()))
    feature_selector_summary = feature_selector_summary.append(
        results, ignore_index=True
    )

    ## validation dataset
    # if validation data is provided...
    if x_valid is not None:
        print("*" * 27)
        print("full validation dataset performance\n")

        y_pred = model.predict(x_valid.values)

        # residual plot
        p = PrettierPlot()
        ax = p.make_canvas(
            title="model: {}\n_parameter set: {}\n_validation data".format(
                model.estimator.__name__, model.model_iter
            ),
            x_label="predicted values",
            y_label="residuals",
            y_shift=0.7,
            title_scale=title_scale,
        )

        p.pretty2d_scatter(
            x=y_pred, y=y_pred - y_valid.values, size=7, color=style.style_grey, ax=ax,
        )
        plt.hlines(
            y=0, xmin=np.min(y_pred), xmax=np.max(y_pred), color=style.style_grey, lw=2
        )
        plt.show()

        # validation data results summary
        y_pred = model.predict(x_valid.values)
        results = self.regression_stats(
            model=model,
            y_True=y_valid.values,
            y_pred=y_pred,
            feature_count=x_train.shape[1],
            data_type="validation",
        )
        feature_selector_summary = feature_selector_summary.append(
            results, ignore_index=True
        )
        display(feature_selector_summary)

    else:
        # if validation data is not provided, then perform k_fold cross validation on
        # training data
        cv = list(
            model_selection.KFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            ).split(x_train, y_train)
        )

        # generate colors
        color_list = style.color_gen(color_map, num=len(cv))

        print("*" * 27)
        print("cross_validation performance\n")

        # residual plot
        p = PrettierPlot(plot_orientation="wide_standard")

        # reshape ubsplot gird
        if n_folds == 2:
            nrows, ncols = 1, 2
        elif n_folds == 3:
            nrows, ncols = 1, 3
        elif n_folds == 4:
            nrows, ncols = 2, 2
        elif n_folds == 5:
            nrows, ncols = 2, 3
        elif n_folds == 6:
            nrows, ncols = 2, 3
        elif n_folds == 7:
            nrows, ncols = 2, 4
        elif n_folds == 8:
            nrows, ncols = 2, 4
        elif n_folds == 9:
            nrows, ncols = 3, 3
        elif n_folds == 10:
            nrows, ncols = 2, 5

        for i, (train_ix, valid_ix) in enumerate(cv):
            x_train_cv = x_train.iloc[train_ix]
            y_train_cv = y_train.iloc[train_ix]
            x_valid_cv = x_train.iloc[valid_ix]
            y_valid_cv = y_train.iloc[valid_ix]

            y_pred = model.fit(x_train_cv.values, y_train_cv.values).predict(
                x_valid_cv.values
            )

            ax = p.make_canvas(
                title="cv fold {}".format(i + 1),
                nrows=nrows,
                ncols=ncols,
                index=i + 1,
                title_scale=title_scale,
            )

            p.pretty2d_scatter(
                x=y_pred,
                y=y_pred - y_valid_cv.values,
                size=7,
                color=color_list[i],
                ax=ax,
            )
            plt.hlines(
                y=0,
                xmin=np.min(y_pred),
                xmax=np.max(y_pred),
                color=style.style_grey,
                lw=2,
            )

            # cv fold results summary
            results = self.regression_stats(
                model=model,
                y_True=y_valid_cv,
                y_pred=y_pred,
                feature_count=x_valid_cv.shape[1],
                data_type="validation",
                fold=i + 1,
            )
            feature_selector_summary = feature_selector_summary.append(
                results, ignore_index=True
            )
        plt.show()
        display(feature_selector_summary)
