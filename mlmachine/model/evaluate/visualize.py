import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn.model_selection as model_selection
import sklearn.metrics as metrics

from prettierplot.plotter import PrettierPlot
from prettierplot import style

import shap


def classificationPanel(self, model, XTrain, yTrain, XValid=None, yValid=None, cmLabels = None, nFolds=3,
                        titleScale=0.7, colorMap="viridis", randomState=1):
    """
    Documentation:
        Description:
            Generate a panel of reports and visualizations summarizing the
            performance of a classification model.
        Paramaters:
            model : model object
                Instantiated model object.
            XTrain : Pandas DataFrame
                Training data observations.
            yTrain : Pandas Series
                Training data labels.
            XValid : Pandas DataFrame, default = None
                Validation data observations.
            yValid : Pandas Series, default = None
                Validation data labels.
            cmLabels : list, default = None
                Custom labels for confusion matrix axes. If left as None,
                will default to 0, 1, 2...
            nFolds : int, default = 3
                Number of cross-validation folds to use when generating
                CV ROC graph.
            colorMap : string specifying built-in matplotlib colormap, default = "viridis"
                Colormap from which to draw plot colors.
            titleScale : float, default = 1.0
                Controls the scaling up (higher value) and scaling down (lower value) of the size of
                the main chart title, the x-axis title and the y-axis title.
            randomState : int, default = 1
                Random number seed.
    """

    print('*' * 55)
    print('* Estimator: {}'.format(model.estimator.__name__))
    print('* Parameter set: {}'.format(model.modelIter))
    print('*' * 55)

    # visualize results with confusion matrix
    p = PrettierPlot(chartProp=18)
    ax = p.makeCanvas(
        title="Model: {}\nParameter set: {}".format(
            model.estimator.__name__, model.modelIter
        ),
        xLabel="Predicted",
        yLabel="Actual",
        yShift=0.4,
        xShift=0.25,
        position=211,
        titleScale=titleScale
    )

    # conditional control for which data is used to generate predictions
    if XValid is not None:
        yPred = model.fit(XTrain, yTrain).predict(XValid)
        print(metrics.classification_report(yValid, yPred, labels=np.unique(yTrain.values)))
        p.prettyConfusionMatrix(
            yTrue=yValid,
            yPred=yPred,
            labels=cmLabels if cmLabels is not None else np.unique(yTrain.values),
            ax=None,
        )
    else:
        yPred = model.fit(XTrain, yTrain).predict(XTrain)
        print(metrics.classification_report(yTrain, yPred, labels=np.unique(yTrain.values)))
        p.prettyConfusionMatrix(
            yTrue=yTrain,
            yPred=yPred,
            labels=cmLabels if cmLabels is not None else np.unique(yTrain.values),
            ax=None,
        )
    plt.show()

    # standard ROC curve for full training dataset or validation dataset
    # if XValid is passed in as None, generate ROC curve for training data
    p = PrettierPlot(chartProp=15)
    ax = p.makeCanvas(
        title="ROC curve - {} data\nModel: {}\nParameter set: {}".format(
            'training' if XValid is None else 'validation',
             model.estimator.__name__,
             model.modelIter
        ),
        xLabel="false positive rate",
        yLabel="true positive rate",
        yShift=0.5,
        position=111 if XValid is not None else 121,
        titleScale=titleScale
    )
    p.prettyRocCurve(
        model=model,
        XTrain=XTrain,
        yTrain=yTrain,
        XValid=XValid,
        yValid=yValid,
        linecolor=style.styleGrey,
        ax=ax,
    )

    # if no validation data is passed, then perform k-fold cross validation and generate
    # an ROC curve for each held out validation set.
    if XValid is None:
        # cross-validated ROC curve
        cv = list(
            model_selection.StratifiedKFold(
                n_splits=nFolds, shuffle=True, random_state=randomState
            ).split(XTrain, yTrain)
        )

        # generate colors
        colorList = style.colorGen(colorMap, num = len(cv))

        # plot ROC curves
        ax = p.makeCanvas(
            title="ROC curve - validation data, {}-fold CV\nModel: {}\nParameter set: {}".format(
                nFolds, model.estimator.__name__, model.modelIter
            ),
            xLabel="false positive rate",
            yLabel="true positive rate",
            yShift=0.5,
            titleScale=titleScale,
            position=122,
        )
        for i, (trainIx, validIx) in enumerate(cv):
            XTrainCV = XTrain.iloc[trainIx]
            yTrainCV = yTrain.iloc[trainIx]
            XValidCV = XTrain.iloc[validIx]
            yValidCV = yTrain.iloc[validIx]

            p.prettyRocCurve(
                model=model,
                XTrain=XTrainCV,
                yTrain=yTrainCV,
                XValid=XValidCV,
                yValid=yValidCV,
                linecolor=colorList[i],
                ax=ax,
            )
    plt.subplots_adjust(wspace=0.3)
    plt.show()


def regressionPanel(self, model, XTrain, yTrain, XValid=None, yValid=None, nFolds=3, titleScale=0.7,
                    colorMap="viridis", randomState=1):
    """
    Documentation:
        Description:
            Creates a set of residual plots and Pandas DataFrames, where each row captures various summary statistics
            pertaining to a model's performance. Generates residual plots and captures performance data for training
            and validation datasets. If no validation set is provided, then cross-validation is performed on the
            training dataset.
        Paramaters:
            model : model object
                Instantiated model object.
            XTrain : Pandas DataFrame
                Training data observations.
            yTrain : Pandas Series
                Training data labels.
            XValid : Pandas DataFrame, default = None
                Validation data observations.
            yValid : Pandas Series, default = None
                Validation data labels.
            nFolds : int, default = 3
                Number of cross-validation folds to use when generating
                CV ROC graph.
            randomState : int, default = 1
                Random number seed.
            titleScale : float, default = 1.0
                Controls the scaling up (higher value) and scaling down (lower value) of the size of
                the main chart title, the x-axis title and the y-axis title.
            colorMap : string specifying built-in matplotlib colormap, default = "viridis"
                Colormap from which to draw plot colors.
    """

    print("*" * 55)
    print("* Estimator: {}".format(model.estimator.__name__))
    print("* Parameter set: {}".format(model.modelIter))
    print("*" * 55 + "\n")

    model.fit(XTrain.values, yTrain.values)

    print('*' * 27)
    print("Full training dataset performance\n")

    ## training dataset
    yPred = model.predict(XTrain.values)

    # residual plot
    p = PrettierPlot()
    ax = p.makeCanvas(
        title="Model: {}\nParameter set: {}\nTraining data".format(
            model.estimator.__name__, model.modelIter
        ),
        xLabel="Predicted values",
        yLabel="Residuals",
        yShift=0.7,
        titleScale=titleScale,
    )

    p.pretty2dScatter(
        x=yPred,
        y=yPred - yTrain.values,
        size=7,
        color=style.styleGrey,
        ax=ax,
    )
    plt.hlines(y=0, xmin=np.min(yPred), xmax=np.max(yPred), color=style.styleGrey, lw=2)
    plt.show()

    # training data results summary
    results = self.regressionStats(model=model,
                              yTrue=yTrain.values,
                              yPred=yPred,
                              featureCount=XTrain.shape[1]
                        )
    # create shell results DataFrame and append
    resultsSummary = pd.DataFrame(columns = list(results.keys()))
    resultsSummary = resultsSummary.append(results, ignore_index=True)

    ## Validation dataset
    # if validation data is provided...
    if XValid is not None:
        print('*' * 27)
        print("Full validation dataset performance\n")

        yPred = model.predict(XValid.values)

        # residual plot
        p = PrettierPlot()
        ax = p.makeCanvas(
            title="Model: {}\nParameter set: {}\nValidation data".format(
                model.estimator.__name__, model.modelIter
            ),
            xLabel="Predicted values",
            yLabel="Residuals",
            yShift=0.7,
            titleScale=titleScale,
        )

        p.pretty2dScatter(
            x=yPred,
            y=yPred - yValid.values,
            size=7,
            color=style.styleGrey,
            ax=ax,
        )
        plt.hlines(y=0, xmin=np.min(yPred), xmax=np.max(yPred), color=style.styleGrey, lw=2)
        plt.show()

        # validation data results summary
        yPred = model.predict(XValid.values)
        results = self.regressionStats(model=model,
                                  yTrue=yValid.values,
                                  yPred=yPred,
                                  featureCount=XTrain.shape[1],
                                  dataType='validation'
                            )
        resultsSummary = resultsSummary.append(results, ignore_index=True)
        display(resultsSummary)

    else:
        # if validation data is not provided, then perform K-fold cross validation on
        # training data
        cv = list(
            model_selection.KFold(
                n_splits=nFolds, shuffle = True, random_state=randomState
            ).split(XTrain, yTrain)
        )

        # generate colors
        colorList = style.colorGen(colorMap, num = len(cv))

        print('*' * 27)
        print("Cross-validation performance\n")

        # residual plot
        p = PrettierPlot(plotOrientation="wideStandard")

        # reshape ubsplot gird
        if nFolds == 2:
            nrows, ncols = 1, 2
        elif nFolds == 3:
            nrows, ncols = 1, 3
        elif nFolds == 4:
            nrows, ncols = 2, 2
        elif nFolds == 5:
            nrows, ncols = 2, 3
        elif nFolds == 6:
            nrows, ncols = 2, 3
        elif nFolds == 7:
            nrows, ncols = 2, 4
        elif nFolds == 8:
            nrows, ncols = 2, 4
        elif nFolds == 9:
            nrows, ncols = 3, 3
        elif nFolds == 10:
            nrows, ncols = 2, 5

        for i, (trainIx, validIx) in enumerate(cv):
            XTrainCV = XTrain.iloc[trainIx]
            yTrainCV = yTrain.iloc[trainIx]
            XValidCV = XTrain.iloc[validIx]
            yValidCV = yTrain.iloc[validIx]

            yPred = model.fit(XTrainCV.values, yTrainCV.values).predict(XValidCV.values)

            ax = p.makeCanvas(
                title="CV fold {}".format(i+1),
                nrows=nrows,
                ncols=ncols,
                index=i+1,
                titleScale=titleScale,
            )

            p.pretty2dScatter(
                x=yPred,
                y=yPred - yValidCV.values,
                size=7,
                color=colorList[i],
                ax=ax,
            )
            plt.hlines(y=0, xmin=np.min(yPred), xmax=np.max(yPred), color=style.styleGrey, lw=2)

            # CV fold results summary
            results = self.regressionStats(model=model,
                                      yTrue=yValidCV,
                                      yPred=yPred,
                                      featureCount=XValidCV.shape[1],
                                      dataType='validation',
                                      fold=i+1
                                )
            resultsSummary = resultsSummary.append(results, ignore_index=True)
        plt.show()
        display(resultsSummary)