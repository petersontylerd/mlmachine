import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn.model_selection as model_selection
import sklearn.metrics as metrics

from prettierplot.plotter import PrettierPlot
from prettierplot import style

import shap


def single_shap_value_tree(self, obs_ix, model, data):
    """
    documentation:
        description:
            generate elements necessary for creating a shap force plot for
            a single observation. works with tree_based models, including:
                - ensemble.random_forest_classifier (package: sklearn)
                - ensemble.gradient_boosting_classifier (package: sklearn)
                - ensemble.extra_trees_classifier (package: sklearn)
                - lightgbm.lgbm_classifier (package: lightgbm)
                - xgboost.xgb_classifier (package: xgboost)
        paramaters:
            obs_ix : int
                index of observation to be analyzed.
            model : model object
                instantiated model object.
            data : pandas DataFrame
                dataset from which to slice indiviudal observation.
        returns:
            obs_data : array
                feature values for the specified observation.
            base_value : float
                expected value associated with positive class.
            obs_shap_values : array
                data array containing the shap values for the specified
                observation.
    """
    # collect observation feature values, model expected value and observation
    # shap values
    obs_data = data.loc[obs_ix].values.reshape(1, _1)
    explainer = shap.TreeExplainer(model.model)
    obs_shap_values = explainer.shap_values(obs_data)

    # different types of models generate differently formatted shap values
    # and expected values
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
    documentation:
        description:
            generate a shap force plot for a single observation.
            works with tree_based models, including:
                - ensemble.random_forest_classifier (package: sklearn)
                - ensemble.gradient_boosting_classifier (package: sklearn)
                - ensemble.extra_trees_classifier (package: sklearn)
                - lightgbm.lgbm_classifier (package: lightgbm)
                - xgboost.xgb_classifier (package: xgboost)
        paramaters:
            obs_ix : int
                index of observations to be visualized.
            model : model object
                instantiated model object.
            data : pandas DataFrame
                dataset from which to slice indiviudal observation.
            target : Pandas Series, default =None
                True labels for observations. this is optional to allow explainations
                for observations without labels.
            classification : boolean, default=True
                boolean argument indicating whether the supervised learning
                task is classification or regression.
            cmap : string, colormap, default = viridis
                colormap to use on force plot.
    """
    # create shap value objects
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
        # shap_values=obs_shap_values,
        # features=obs_data,
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
    documentation:
        description:
            generate elements necessary for creating a shap force plot for
            multiple observations simultaneously. works with tree_based
            models, including:
                - ensemble.random_forest_classifier (package: sklearn)
                - ensemble.gradient_boosting_classifier (package: sklearn)
                - ensemble.extra_trees_classifier (package: sklearn)
                - lightgbm.lgbm_classifier (package: lightgbm)
                - xgboost.xgb_classifier (package: xgboost)
        paramaters:
            obs_ixs : array or list
                index values of observations to be analyzed.
            model : model object
                instantiated model object.
            data : pandas DataFrame
                dataset from which to slice indiviudal observations.
        returns:
            obs_data : array
                feature values for the specified observations.
            base_value : float
                expected value associated with positive class.
            obs_shap_values : array
                data array containing the shap values for the specified
                observations.
    """
    obs_data = data.loc[obs_ixs].values
    explainer = shap.TreeExplainer(model.model)
    obs_shap_values = explainer.shap_values(obs_data)

    #
    if isinstance(obs_shap_values, list):
        obs_shap_values = obs_shap_values[1]
    else:
        obs_shap_values = obs_shap_values

    #
    if isinstance(explainer.expected_value, np.floating):
        base_value = explainer.expected_value
    else:
        base_value = explainer.expected_value[1]

    return obs_data, base_value, obs_shap_values


def multi_shap_viz_tree(self, obs_ixs, model, data):
    """
    documentation:
        description:
            generate a shap force plot for multiple  observations simultaneously.
            works with tree_based models, including:
                - ensemble.random_forest_classifier (package: sklearn)
                - ensemble.gradient_boosting_classifier (package: sklearn)
                - ensemble.extra_trees_classifier (package: sklearn)
                - lightgbm.lgbm_classifier (package: lightgbm)
                - xgboost.xgb_classifier (package: xgboost)
        paramaters:
            obs_ixs : array or list
                index values of observations to be analyzed.
            model : model object
                instantiated model object.
            data : pandas DataFrame
                dataset from which to slice observations.
    """
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
    documentation:
        description:
            generate a shap dependence plot for a pair of features. one feature is
            represented in a scatter plot, with each observation's actual value on the
            x_axis and the corresponding shap value for the observation on the y_axis.
            the second feature applies a hue to the scattered feature based on the
            individual values of the second feature.
        paramaters:
            obs_data : array
                feature values for the specified observations.
            obs_shap_values : array
                data array containing the shap values for the specified
                observations.
            scatter_feature : string
                name of feature to scatter on plot area.
            color_feature : string
                name of feature to apply color to dots in scatter plot.
            feature_names : list
                list of all feature names in the dataset.
            x_jitter : float, default = 0.08
                controls displacement of dots along x_axis.
            dot_size : float, default = 25
                size of dots.
            alpha : float, default = 0.7
                transparency of dots.
            ax : axes object, default =None
                axis on which to place visual.
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


def shap_dependence_grid(self, obs_data, obs_shap_values, grid_features, all_features, dot_size, alpha):
    """
    documentation:
        description:
            generate a shap dependence plot for a pair of features. one feature is
            represented in a scatter plot, with each observation's actual value on the
            x_axis and the corresponding shap value for the observation on the y_axis.
            the second feature applies a hue to the scattered feature based on the
            individual values of the second feature.
        paramaters:
            obs_data : array
                feature values for the specified observations.
            obs_shap_values : array
                data array containing the shap values for the specified
                observations.
            grid_features : list
                names of features to display on grid.
            all_features : list
                list containing names for all features for which shap values were
                calculated.
            dot_size : float, default = 25
                size of dots.
            alpha : float, default = 0.7
                transparency of dots.
    """

    fig, ax = plt.subplots(
        ncols=len(grid_features),
        nrows=len(grid_features),
        constrained_layout=True,
        figsize=(len(grid_features) * 3.5, len(grid_features) * 2.5),
    )

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
    documentation:
        description:
            generate a shap summary plot for multiple  observations.
        paramaters:
            obs_data : array
                feature values for the specified observations.
            obs_shap_values : array
                data array containing the shap values for the specified
                observations.
            feature_names : list
                list of all feature names in the dataset.
            alpha : float, default = 0.7
                controls transparency of dots.
    """
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