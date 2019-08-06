import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn.model_selection as model_selection
import sklearn.metrics as metrics

from prettierplot.plotter import PrettierPlot
from prettierplot import style

import shap


def classificationPanel(self, model, nFolds=3, randomState=1):
    """
    Documentation:
        Description:
            Generate a panel of reports and visualizations summarizing the
            performance of a classification model.
        Paramaters:
            model : model object
                Instantiated model object.
            nFols : int, default = 3
                Number of cross-validation folds to use when generating
                CV ROC graph.
            randomState : int, default = 1
                Random number seed.
    """

    model.fit(self.data, self.target)
    yPred = model.predict(self.data)

    print('*' * 55)
    print('* Estimator: {}'.format(model.estimator.split(".")[1]))
    print('* Parameter set: {}'.format(model.modelIter))
    print('*' * 55)

    # generate simple classification report
    print(metrics.classification_report(self.target, yPred, labels=[0, 1]))

    # visualize results with confusion matrix
    p = PrettierPlot()
    ax = p.makeCanvas(
        title="Model: {}\nParameter set: {}".format(
            model.estimator.split(".")[1], model.modelIter
        ),
        xLabel="Predicted",
        yLabel="Actual",
        yShift=0.5,
        xShift=0.35,
        position=211,
    )
    p.prettyConfusionMatrix(
        yTrue=self.target, yPred=yPred, labels=["Survived", "Died"], ax=None
    )
    plt.show()

    # standard ROC curve
    p = PrettierPlot(chartProp=15)
    ax = p.makeCanvas(
        title="Model: {}\nParameter set: {}".format(
            model.estimator.split(".")[1], model.modelIter
        ),
        xLabel="false positive rate",
        yLabel="true positive rate",
        yShift=0.43,
        position=121,
    )
    p.prettyRocCurve(
        model=model,
        XTrain=self.data,
        yTrain=self.target,
        linecolor=style.styleHexMid[0],
        ax=ax,
    )

    # cross-validated ROC curve
    cv = list(
        model_selection.StratifiedKFold(
            n_splits=nFolds, random_state=randomState
        ).split(self.data, self.target)
    )

    # plot ROC curves
    ax = p.makeCanvas(
        title="Model: {}\nParameter set: {}".format(
            model.estimator.split(".")[1], model.modelIter
        ),
        xLabel="false positive rate",
        yLabel="true positive rate",
        yShift=0.43,
        position=122,
    )
    for i, (trainIx, _) in enumerate(cv):
        XTrainCV = self.data.iloc[trainIx]
        yTrainCV = self.target.iloc[trainIx]

        p.prettyRocCurve(
            model=model,
            XTrain=XTrainCV,
            yTrain=yTrainCV,
            linecolor=style.styleHexMid[i],
            ax=ax,
        )
    plt.show()

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
    obsData = data.iloc[obsIx].values.reshape(1, -1)
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


def singleShapVizTree(self, obsIx, model, data, target=None):
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
    """
    # create SHAP value objects
    obsData, baseValue, obsShapValues = self.singleShapValueTree(obsIx=obsIx, model=model, data=data)
    
    # display summary information about prediction
    probas = model.predict_proba(obsData)
    print('Negative class probability: {:.6f}'.format(probas[0][0]))
    print('Positive class probability: {:.6f}'.format(probas[0][1]))
    
    if target is not None:
        print('True label: {}'.format(target[obsIx]))
    
    # display force plot
    shap.force_plot(
        base_value=baseValue,
        shap_values=obsShapValues,
        features=obsData,
        feature_names=data.columns.tolist(),
        matplotlib=True,
        show=False
    )
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.grid(b=False)
    plt.show()


# multiple predictions
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
    obsData = data.iloc[obsIxs].values
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
    

def shapDependencePlot(self, obsData, obsShapValues, scatterFeature, colorFeature, featureNames, xJitter=0.08):
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
        )
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.grid(b=False)
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

# def shapValsKernel(model, nSamples):

#     obsData = train.data
    
#     # create object for calculating shap values
#     explainer = shap.KernelExplainer(model.predict_proba, data=obsData, link='logit')

#     obsShapValues = explainer.shap_values(obsData, nsamples=nSamples)
    
#     return explainer, obsShapValues

# explainer, obsShapValues = shapValsKernel(model=modelS, nSamples=10)

# def shapValsKernelViz(obsIx, explainer, obsShapValues, nSummaryDisplay = 5, featureNames=featureNames):
#     obsData = train.data
#     obs = obsData.iloc[obsIx].values.reshape(1,-1)

#     probas = model.predict_proba(obs)
    
#     print('Negative class probability: {:.6f}'.format(probas[0][0]))
#     print('Positive class probability: {:.6f}'.format(probas[0][1]))
#     print('True label: {}'.format(train.target[obsIx]))

#     print(probas)
#     print(np.argmax(probas))


#     # shap value summary
#     shapSummary = pd.DataFrame(
#                     obsShapValues[0][obsIx,:],
#                     index = featureNames,
#                     columns = ['Top SHAP values']
#                 )
#     display(shapSummary.sort_values(['Top SHAP values'], ascending = True if np.argmax(probas) == 1 else False)[:nSummaryDisplay])

#     # display force plot
#     shap.force_plot(
#         base_value = explainer.expected_value[0],
#         shap_values =obsShapValues[0][obsIx,:],
#         features = obs,
#         feature_names = featureNames,
#         matplotlib=True,
#         show=False
#     )
#     plt.rcParams['axes.facecolor'] = 'white'
#     plt.grid(b=False)
#     plt.show()

# shapValsKernelViz(obsIx=0, explainer=explainer, obsShapValues=obsShapValues, nSummaryDisplay = 5, featureNames=featureNames)
