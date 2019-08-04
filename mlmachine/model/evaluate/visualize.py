import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn.model_selection as model_selection
import sklearn.metrics as metrics

from prettierplot.plotter import PrettierPlot
from prettierplot import style


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
    for i, (trainIx, testIx) in enumerate(cv):
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