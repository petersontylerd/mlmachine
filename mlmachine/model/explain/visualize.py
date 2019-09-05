import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn.model_selection as model_selection
import sklearn.metrics as metrics

from prettierplot.plotter import PrettierPlot
from prettierplot import style

import shap


def singleShapValueTree(self, obsIx, model, data):
    """
    Documentation:
        Description:
            Generate elements necessary for creating a SHAP force plot for
            a single observation. Works with tree-based models, including:
                - ensemble.RandomForestClassifier (package: sklearn)
                - ensemble.GradientBoostingClassifier (package: sklearn)
                - ensemble.ExtraTreesClassifier (package: sklearn)
                - lightgbm.LGBMClassifier (package: lightgbm)
                - xgboost.XGBClassifier (package: xgboost)
        Paramaters:
            obsIx : int
                Index of observation to be analyzed.
            model : model object
                Instantiated model object.
            data : Pandas DataFrame
                Dataset from which to slice indiviudal observation.
        Returns:
            obsData : array
                Feature values for the specified observation.
            baseValue : float
                Expected value associated with positive class.
            obsShapValues : array
                Data array containing the SHAP values for the specified
                observation.
    """
    # collect observation feature values, model expected value and observation
    # SHAP values
    obsData = data.loc[obsIx].values.reshape(1, -1)
    explainer = shap.TreeExplainer(model.model)
    obsShapValues = explainer.shap_values(obsData)

    # different types of models generate differently formatted SHAP values
    # and expected values
    if isinstance(obsShapValues, list):
        obsShapValues = obsShapValues[1]
    else:
        obsShapValues = obsShapValues

    if isinstance(explainer.expected_value, np.floating):
        baseValue = explainer.expected_value
    else:
        baseValue = explainer.expected_value[1]

    return obsData, baseValue, obsShapValues


def singleShapVizTree(self, obsIx, model, data, target=None, classification=True):
    """
    Documentation:
        Description:
            Generate a SHAP force plot for a single observation.
            Works with tree-based models, including:
                - ensemble.RandomForestClassifier (package: sklearn)
                - ensemble.GradientBoostingClassifier (package: sklearn)
                - ensemble.ExtraTreesClassifier (package: sklearn)
                - lightgbm.LGBMClassifier (package: lightgbm)
                - xgboost.XGBClassifier (package: xgboost)
        Paramaters:
            obsIx : int
                Index of observations to be visualized.
            model : model object
                Instantiated model object.
            data : Pandas DataFrame
                Dataset from which to slice indiviudal observation.
            target : Pandas Series, default = None
                True labels for observations. This is optional to allow explainations
                for observations without labels.
            classification : boolean, default = True
                Boolean argument indicating whether the supervised learning
                task is classification or regression.
    """
    # create SHAP value objects
    obsData, baseValue, obsShapValues = self.singleShapValueTree(obsIx=obsIx, model=model, data=data)

    # display summary information about prediction
    if classification:
        probas = model.predict_proba(obsData)
        print('Negative class probability: {:.6f}'.format(probas[0][0]))
        print('Positive class probability: {:.6f}'.format(probas[0][1]))

        if target is not None:
            print('True label: {}'.format(target.loc[obsIx]))
    else:
        print('Prediction: {:.6f}'.format(model.predict(obsData)[0]))
        if target is not None:
            print('True label: {:.6f}'.format(target.loc[obsIx]))

    # display force plot
    shap.force_plot(
        base_value=baseValue,
        # shap_values=obsShapValues,
        # features=obsData,
        shap_values=np.around(obsShapValues.astype(np.double),3),
        features=np.around(obsData.astype(np.double),3),
        feature_names=data.columns.tolist(),
        matplotlib=True,
        show=False
    )
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.grid(b=False)
    plt.show()


def multiShapValueTree(self, obsIxs, model, data):
    """
    Documentation:
        Description:
            Generate elements necessary for creating a SHAP force plot for
            multiple observations simultaneously. Works with tree-based
            models, including:
                - ensemble.RandomForestClassifier (package: sklearn)
                - ensemble.GradientBoostingClassifier (package: sklearn)
                - ensemble.ExtraTreesClassifier (package: sklearn)
                - lightgbm.LGBMClassifier (package: lightgbm)
                - xgboost.XGBClassifier (package: xgboost)
        Paramaters:
            obsIxs : array or list
                Index values of observations to be analyzed.
            model : model object
                Instantiated model object.
            data : Pandas DataFrame
                Dataset from which to slice indiviudal observations.
        Returns:
            obsData : array
                Feature values for the specified observations.
            baseValue : float
                Expected value associated with positive class.
            obsShapValues : array
                Data array containing the SHAP values for the specified
                observations.
    """
    obsData = data.loc[obsIxs].values
    explainer = shap.TreeExplainer(model.model)
    obsShapValues = explainer.shap_values(obsData)

    #
    if isinstance(obsShapValues, list):
        obsShapValues = obsShapValues[1]
    else:
        obsShapValues = obsShapValues

    #
    if isinstance(explainer.expected_value, np.floating):
        baseValue = explainer.expected_value
    else:
        baseValue = explainer.expected_value[1]

    return obsData, baseValue, obsShapValues


def multiShapVizTree(self, obsIxs, model, data):
    """
    Documentation:
        Description:
            Generate a SHAP force plot for multiple  observations simultaneously.
            Works with tree-based models, including:
                - ensemble.RandomForestClassifier (package: sklearn)
                - ensemble.GradientBoostingClassifier (package: sklearn)
                - ensemble.ExtraTreesClassifier (package: sklearn)
                - lightgbm.LGBMClassifier (package: lightgbm)
                - xgboost.XGBClassifier (package: xgboost)
        Paramaters:
            obsIxs : array or list
                Index values of observations to be analyzed.
            model : model object
                Instantiated model object.
            data : Pandas DataFrame
                Dataset from which to slice observations.
    """
    obsData, baseValue, obsShapValues = self.multiShapValueTree(obsIxs=obsIxs, model=model, data=data)

    # generate force plot
    visual = shap.force_plot(
        base_value = baseValue,
        shap_values = obsShapValues,
        features = obsData,
        feature_names = data.columns.tolist(),
    )
    return visual


def shapDependencePlot(self, obsData, obsShapValues, scatterFeature, colorFeature, featureNames,
                        xJitter=0.08, dotSize=25, alpha=0.7, show=True, ax=None):
    """
    Documentation:
        Description:
            Generate a SHAP dependence plot for a pair of features. One feature is
            represented in a scatter plot, with each observation's actual value on the
            x-axis and the corresponding SHAP value for the observation on the y-axis.
            The second feature applies a hue to the scattered feature based on the
            individual values of the second feature.
        Paramaters:
            obsData : array
                Feature values for the specified observations.
            obsShapValues : array
                Data array containing the SHAP values for the specified
                observations.
            scatterFeature : string
                Name of feature to scatter on plot area.
            colorFeature : string
                Name of feature to apply color to dots in scatter plot.
            featureNames : list
                List of all feature names in the dataset.
            xJitter : float, default = 0.08
                Controls displacement of dots along x-axis.
            dotSize : float, default = 25
                Size of dots.
            alpha : float, default = 0.7
                Transparency of dots.
            ax : Axes object, default = None
                Axis on which to place visual.
    """

    # generate force plot
    shap.dependence_plot(
            ind = scatterFeature,
            shap_values = obsShapValues,
            features = obsData,
            feature_names = featureNames,
            interaction_index = colorFeature,
            show=False,
            x_jitter=xJitter,
            dot_size=dotSize,
            alpha=alpha,
            ax=ax,
        )
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.grid(b=False)

    if show:
        plt.show()


def shapDependenceGrid(self, obsData, obsShapValues, gridFeatures, allFeatures, dotSize, alpha):
    """
    Documentation:
        Description:
            Generate a SHAP dependence plot for a pair of features. One feature is
            represented in a scatter plot, with each observation's actual value on the
            x-axis and the corresponding SHAP value for the observation on the y-axis.
            The second feature applies a hue to the scattered feature based on the
            individual values of the second feature.
        Paramaters:
            obsData : array
                Feature values for the specified observations.
            obsShapValues : array
                Data array containing the SHAP values for the specified
                observations.
            gridFeatures : list
                Names of features to display on grid.
            allFeatures : list
                List containing names for all features for which SHAP values were
                calculated.
            dotSize : float, default = 25
                Size of dots.
            alpha : float, default = 0.7
                Transparency of dots.
    """

    fig, ax = plt.subplots(
        ncols=len(gridFeatures),
        nrows=len(gridFeatures),
        constrained_layout=True,
        figsize=(len(gridFeatures) * 3.5, len(gridFeatures) * 2.5),
    )

    for colIx, colFeature in enumerate(gridFeatures):
        for rowIx, rowFeature in enumerate(gridFeatures):
            self.shapDependencePlot(
                obsData=obsData,
                obsShapValues=obsShapValues,
                scatterFeature=colFeature,
                colorFeature=rowFeature,
                featureNames=allFeatures,
                dotSize=dotSize,
                alpha=alpha,
                show=False,
                ax=ax[rowIx, colIx],
            )
            ax[rowIx, colIx].yaxis.set_visible(False)
            if rowIx != colIx:
                ax[rowIx, colIx].set_xlabel("{},\nby {}".format(colFeature, rowFeature))
    plt.show()


def shapSummaryPlot(self, obsData, obsShapValues, featureNames, alpha=0.7):
    """
    Documentation:
        Description:
            Generate a SHAP summary plot for multiple  observations.
        Paramaters:
            obsData : array
                Feature values for the specified observations.
            obsShapValues : array
                Data array containing the SHAP values for the specified
                observations.
            featureNames : list
                List of all feature names in the dataset.
            alpha : float, default = 0.7
                Controls transparency of dots.
    """
    shap.summary_plot(
        shap_values = obsShapValues,
        features = obsData,
        feature_names = featureNames,
        alpha = 0.7,
        show=False,
    )
    plt.rcParams['axes.facecolor'] = 'white'
    plt.grid(b=False)
    plt.show()