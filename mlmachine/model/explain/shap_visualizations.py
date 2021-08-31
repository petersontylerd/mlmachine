import os
import pickle
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


def load_shap_objects(self, estimator_name):
    """

    """
    with open(os.path.join(self.shap_explainers_object_dir, f"{estimator_name}_TreeExplainer.pkl"), "rb") as handle:
        explainer = pickle.load(handle)

    with open(os.path.join(self.shap_values_object_dir, f"{estimator_name}_TreeExplainer_training.pkl"), "rb") as handle:
        training_shap_values = pickle.load(handle)

    with open(os.path.join(self.shap_values_object_dir, f"{estimator_name}_TreeExplainer_validation.pkl"), "rb") as handle:
        validation_shap_values = pickle.load(handle)

    return explainer, training_shap_values, validation_shap_values


# def single_shap_value_tree(self, observation_index, model, explainer, shap_values, training_data=True):
def single_shap_value_tree(self, observation_index, model):
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
            observation_index : int
                Index of observation to analyze.
            model : model object
                Fitted model object.
            training_data : boolean, dafault=True
                Controls which dataset (training or validation) is used for visualization.

        ---
        Returns:
            observation_values : array
                Feature values for the specified observation.
            observation_shap_values : array
                Data array containing the SHAP values for the specified
                observation.
            base_value : float
                Expected prediction value.
    """
    # load shap objects for estimator
    estimator_name = model.custom_model.__class__.__name__
    explainer, training_shap_values, validation_shap_values = self.load_shap_objects(estimator_name=estimator_name)

    if observation_index in self.training_features.index:
        training_data = True
    elif observation_index in self.validation_features.index:
        training_data = False
    else:
        raise ValueError("invalid index value provided")
    #
    data, _, _ = self.training_or_validation_dataset(training_data)

    #
    observation_values = data.loc[observation_index].values.reshape(1, -1)

    #
    if training_data:
        observation_shap_values = training_shap_values.loc[observation_index]
    else:
        observation_shap_values = validation_shap_values.loc[observation_index]

    #
    if isinstance(explainer.expected_value, np.floating):
        base_value = explainer.expected_value
    elif isinstance(explainer.expected_value[0], np.floating) and len(explainer.expected_value) == 1:
        base_value = explainer.expected_value
    else:
        base_value = explainer.expected_value[1]

    return observation_values, observation_shap_values, base_value


def single_shap_viz_tree(self, observation_index, model, target=None, classification=True, cmap="viridis"):
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
            observation_index : int
                Index of observation to analyze.
            model : model object
                Fitted model object.
            target : Pandas Series, default=None
                True label for observation.
            classification : bool, default=True
                Bool argument indicating whether the supervised learning task is classification
                task or regression task.
            cmap : str, colormap, default=viridis
                Colormap applied to plot.
            training_data : boolean, dafault=True
                Controls which dataset (training or validation) is used for visualization.
    """
    # return observation features values, expected value, and observation SHAP values
    observation_values, observation_shap_values, base_value =\
         self.single_shap_value_tree(
                    observation_index=observation_index,
                    model=model,
                )

    # display summary information about prediction
    if classification:
        probas = model.predict_proba(observation_values)
        print('negative class probability: {:.6f}'.format(probas[0][0]))
        print('positive class probability: {:.6f}'.format(probas[0][1]))

        if target is not None:
            print('True label: {}'.format(target.loc[observation_index]))
    else:
        print('prediction: {:.6f}'.format(model.predict(observation_values)[0]))
        if target is not None:
            print('True label: {:.6f}'.format(target.loc[observation_index]))

    # display force plot
    shap.force_plot(
        base_value=base_value,
        shap_values=np.around(observation_shap_values.astype(np.double),3).values,
        features=np.around(observation_values.astype(np.double),3),
        feature_names=self.training_features.columns.tolist(),
        matplotlib=True,
        show=False,
        plot_cmap=cmap,
        text_rotation=20,
    )
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.grid(b=False)
    plt.show()


def multi_shap_value_tree(self, observation_indexes, model):
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

    # load shap objects for estimator
    estimator_name = model.custom_model.__class__.__name__
    explainer, training_shap_values, validation_shap_values = self.load_shap_objects(estimator_name=estimator_name)

    if set(observation_indexes).issubset(self.training_features.index):
        training_data = True
    elif set(observation_indexes).issubset(self.validation_features.index):
        training_data = False
    else:
        raise ValueError("Invalid index value detected.")
    #
    data, _, _ = self.training_or_validation_dataset(training_data)

    # collect observation feature values, explainer object and observation SHAP values
    observation_values = data.loc[observation_indexes].values

    #
    if training_data:
        observation_shap_values = training_shap_values.loc[observation_indexes]
    else:
        observation_shap_values = validation_shap_values.loc[observation_indexes]

    #
    if isinstance(explainer.expected_value, np.floating):
        base_value = explainer.expected_value
    else:
        base_value = explainer.expected_value[1]

    return observation_values, observation_shap_values, base_value


def multi_shap_viz_tree(self, observation_indexes, model):
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
            observation_indexes : array or list
                Index values of observations to analyze.
            model : model object
                Instantiated model object.
            data : Pandas DataFrame
                Dataset from which to slice observations' feature values.
    """
    # return observation features values, expected value, and observation SHAP values
    observation_values, observation_shap_values, base_value = \
        self.multi_shap_value_tree(
                    observation_indexes=observation_indexes,
                    model=model,
                )

    # generate force plot
    visual = shap.force_plot(
        base_value = base_value,
        shap_values = observation_shap_values.values,
        features = observation_values,
        feature_names = self.training_features.columns.tolist(),
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