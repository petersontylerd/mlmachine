import numpy as np
import pandas as pd
import sklearn.feature_selection as feature_selection


def featureImportanceSummary(self, k=5):
    """
    Documentation:
        Description:
            Create table that summarizes the importance of each feature. Statistics include
            F-score, P-value and support.
        Parameters:
            k : int, default = 5
                Identify the top k features based on feature importance.
        Returns:
            Pandas DataFrame containing summary data describing feature i.mportance
    """
    # create SelectKBest regression
    featSelector = feature_selection.SelectKBest(feature_selection.f_regression, k=k)
    _ = featSelector.fit(self.data, self.target)

    # build dataframe
    featScores = pd.DataFrame()
    featScores["F score"] = featSelector.scores_
    featScores["P value"] = featSelector.pvalues_
    featScores["Support"] = featSelector.get_support()
    featScores["Attribute"] = self.data.columns
    return featScores.sort_values(["F score"], ascending=[False])

