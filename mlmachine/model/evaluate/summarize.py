import numpy as np
import pandas as pd


def topBayesOptimModels(self, bayesOptimSummary, numModels):
    """
    Documentation:
        Description:

        Paramaters:

        Returns:

    """

    models = {}
    for estimator in bayesOptimSummary["estimator"].unique():
        estDf = bayesOptimSummary[bayesOptimSummary["estimator"] == estimator].sort_values(
            ["mean"], ascending=[False]
        )["iteration"][:numModels]
        models[estimator] = estDf.values.tolist()
    return models
