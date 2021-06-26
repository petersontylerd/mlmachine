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
)

from prettierplot.plotter import PrettierPlot
from prettierplot import style

import shap


def single_shap_value_tree(self, obs_ix, model, data):
    """
    Documentation:

        ---
        Description:
            Generate elements necessary for creating a SHAP force plot for a single
            observation. Works with tree-based models, including:
                - RandomForestClassifier (package: sklearn)
                - GradientBoostingClassifier (package: sklearn)
                - ExtraTreesClassifier (package: sklearn)
                - LGBMClassifier (package: lightgbm)
                - XGBClassifier (package: xgboost)

        ---
        Parameters:
            obs_ix : int
                Index of observation to analyze.
            model : model object
                Instantiated model object.
            data : Pandas DataFrame
                Dataset from which to slice individual observation's features.

        ---
        Returns:
            obs_data : array
                Feature values for the specified observation.
            base_value : float
                Expected prediction value.
            obs_shap_values : array
                Data array containing the SHAP values for the specified
                observation.
    """
    # collect observation feature values, explainer object and observation SHAP values
    obs_data = data.loc[obs_ix].values.reshape(1, -1)
    explainer = shap.TreeExplainer(model.custom_model)
    obs_shap_values = explainer.shap_values(obs_data)

    # accommodate the fact that different types of models generate differently
    # formatted SHAP values and expected values
    if isinstance(obs_shap_values, list):
        obs_shap_values = obs_shap_values[1]
    else:
        obs_shap_values = obs_shap_values

    if isinstance(explainer.expected_value, np.floating):
        base_value = explainer.expected_value
    else:
        base_value = explainer.expected_value[1]

    return obs_data, base_value, obs_shap_values


def single_shap_viz_tree(self, obs_ix, model, data, target=None, classification=True, cmap="viridis"):
    """
    Documentation:

        ---
        Description:
            Generate a SHAP force plot for a single observation. Works with tree-based models,
            including:
                - RandomForestClassifier (package: sklearn)
                - GradientBoostingClassifier (package: sklearn)
                - ExtraTreesClassifier (package: sklearn)
                - LGBMClassifier (package: lightgbm)
                - XGBClassifier (package: xgboost)

        ---
        Parameters:
            obs_ix : int
                Index of observation to visualize.
            model : model object
                Instantiated model object.
            data : Pandas DataFrame
                Dataset from which to slice indiviudal observation's features.
            target : Pandas Series, default=None
                True label for observation.
            classification : bool, default=True
                Bool argument indicating whether the supervised learning task is classification
                task or regression task.
            cmap : str, colormap, default=viridis
                Colormap applied to plot.
    """
    # return observation features values, expected value, and observation SHAP values
    obs_data, base_value, obs_shap_values = self.single_shap_value_tree(obs_ix=obs_ix, model=model, data=data)

    # display summary information about prediction
    if classification:
        probas = model.predict_proba(obs_data)
        print('negative class probability: {:.6f}'.format(probas[0][0]))
        print('positive class probability: {:.6f}'.format(probas[0][1]))

        if target is not None:
            print('True label: {}'.format(target.loc[obs_ix]))
    else:
        print('prediction: {:.6f}'.format(model.predict(obs_data)[0]))
        if target is not None:
            print('True label: {:.6f}'.format(target.loc[obs_ix]))

    # display force plot
    shap.force_plot(
        base_value=base_value,
        shap_values=np.around(obs_shap_values.astype(np.double),3),
        features=np.around(obs_data.astype(np.double),3),
        feature_names=data.columns.tolist(),
        matplotlib=True,
        show=False,
        plot_cmap=cmap
    )
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.grid(b=False)
    plt.show()


def multi_shap_value_tree(self, obs_ixs, model, data):
    """
    Documentation:

        ---
        Description:
            Generate elements necessary for creating a SHAP force plot for multiple
            observations simultaneously. Works with tree-based models, including:
                - RandomForestClassifier (package: sklearn)
                - GradientBoostingClassifier (package: sklearn)
                - ExtraTreesClassifier (package: sklearn)
                - LGBMClassifier (package: lightgbm)
                - XGBClassifier (package: xgboost)

        ---
        Parameters:
            obs_ixs : array or list
                Index values of observation to be analyze.
            model : model object
                Instantiated model object.
            data : Pandas DataFrame
                Dataset from which to slice indiviudal observations' feature values.

        ---
        Returns:
            obs_data : array
                Feature values for the specified observations.
            base_value : float
                Expected predicted value.
            obs_shap_values : array
                data array containing the SHAP values for the specified
                observations.
    """

    # collect observation feature values, explainer object and observation SHAP values
    obs_data = data.loc[obs_ixs].values
    explainer = shap.TreeExplainer(model.custom_model)
    obs_shap_values = explainer.shap_values(obs_data)

    # accommodate the fact that different types of models generate differently
    # formatted SHAP values and expected values
    if isinstance(obs_shap_values, list):
        obs_shap_values = obs_shap_values[1]
    else:
        obs_shap_values = obs_shap_values

    if isinstance(explainer.expected_value, np.floating):
        base_value = explainer.expected_value
    else:
        base_value = explainer.expected_value[1]

    return obs_data, base_value, obs_shap_values


def multi_shap_viz_tree(self, obs_ixs, model, data):
    """
    Documentation:

        ---
        Description:
            Generate a SHAP force plot for multiple observations simultaneously. Works with
            tree-based models, including:
                - RandomForestClassifier (package: sklearn)
                - GradientBoostingClassifier (package: sklearn)
                - ExtraTreesClassifier (package: sklearn)
                - LGBMClassifier (package: lightgbm)
                - XGBClassifier (package: xgboost)

        ---
        Parameters:
            obs_ixs : array or list
                Index values of observations to analyze.
            model : model object
                Instantiated model object.
            data : Pandas DataFrame
                Dataset from which to slice observations' feature values.
    """
    # return observation features values, expected value, and observation SHAP values
    obs_data, base_value, obs_shap_values = self.multi_shap_value_tree(obs_ixs=obs_ixs, model=model, data=data)

    # generate force plot
    visual = shap.force_plot(
        base_value = base_value,
        shap_values = obs_shap_values,
        features = obs_data,
        feature_names = data.columns.tolist(),
    )
    return visual


def shap_dependence_plot(self, obs_data, obs_shap_values, scatter_feature, color_feature, feature_names,
                        x_jitter=0.08, dot_size=25, alpha=0.7, show=True, ax=None):
    """
    Documentation:

        ---
        Description:
            Generate a SHAP dependence plot for a pair of features. One feature is
            represented in a scatter plot, with each observation's actual value on the
            x-axis and the corresponding SHAP value for the observation on the y-axis.
            the second feature applies a hue to the scattered feature based on the
            individual values of the second feature.

        ---
        Parameters:
            obs_data : array
                Feature values for the specified observations.
            obs_shap_values : array
                Data array containing the SHAP values for the specified observations.
            scatter_feature : str
                Name of feature to scatter on plot area.
            color_feature : str
                Name of feature to use when applying color to dots on scatter plot.
            feature_names : list
                List of all feature names in the dataset.
            x_jitter : float, default=0.08
                Controls displacement of dots along x-axis.
            dot_size : float, default=25
                Size of dots.
            alpha : float, default=0.7
                Transparency of dots.
            show : boolean, default=True
                Conditional controlling whether to display plot automatically when
                calling function.
            ax : axes object, default=None
                Axis on which to place visual.
    """
    # generate force plot
    shap.dependence_plot(
            ind = scatter_feature,
            shap_values = obs_shap_values,
            features = obs_data,
            feature_names = feature_names,
            interaction_index = color_feature,
            show=False,
            x_jitter=x_jitter,
            dot_size=dot_size,
            alpha=alpha,
            ax=ax,
        )
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.grid(b=False)

    if show:
        plt.show()


def shap_dependence_grid(self, obs_data, obs_shap_values, grid_features, all_features, dot_size=25, alpha=0.7):
    """
    Documentation:
        Description:
            Generate a SHAP dependence plot for a pair of features. One feature is
            represented in a scatter plot, with each observation's actual value on the
            x-axis and the corresponding SHAP value for the observation on the y-axis.
            The second feature applies a hue to the scattered feature based on the
            individual values of the second feature.
        Parameters:
            obs_data : array
                feature values for the specified observations.
            obs_shap_values : array
                Data array containing the SHAP values for the specified
                observations.
            grid_features : list
                Names of features to display on grid.
            all_features : list
                List containing names for all features for which SHAP values were
                calculated.
            dot_size : float, default=25
                Size of dots.
            alpha : float, default=0.7
                Transparency of dots.
    """
    # create subplot grid
    fig, ax = plt.subplots(
        ncols=len(grid_features),
        nrows=len(grid_features),
        constrained_layout=True,
        figsize=(len(grid_features) * 3.5, len(grid_features) * 2.5),
    )

    # iterate through column feature and row feature pairs
    for col_ix, col_feature in enumerate(grid_features):
        for row_ix, row_feature in enumerate(grid_features):
            self.shap_dependence_plot(
                obs_data=obs_data,
                obs_shap_values=obs_shap_values,
                scatter_feature=col_feature,
                color_feature=row_feature,
                feature_names=all_features,
                dot_size=dot_size,
                alpha=alpha,
                show=False,
                ax=ax[row_ix, col_ix],
            )
            ax[row_ix, col_ix].yaxis.set_visible(False)
            if row_ix != col_ix:
                ax[row_ix, col_ix].set_xlabel("{},\nby {}".format(col_feature, row_feature))
    plt.show()


def shap_summary_plot(self, obs_data, obs_shap_values, feature_names, alpha=0.7):
    """
    Documentation:
        Description:
            Generate a SHAP summary plot for multiple observations.
        Parameters:
            obs_data : array
                Feature values for the specified observations.
            obs_shap_values : array
                Data array containing the SHAP values for the specified
                observations.
            feature_names : list
                List of all feature names in the dataset.
            alpha : float, default=0.7
                Controls transparency of dots.
    """
    # generate SHAP summary plot
    shap.summary_plot(
        shap_values = obs_shap_values,
        features = obs_data,
        feature_names = feature_names,
        alpha = 0.7,
        show=False,
    )
    plt.rcParams['axes.facecolor'] = 'white'
    plt.grid(b=False)
    plt.show()