
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

def selectKBestRef(dataFrame, target, k = 5):
    featSelector = SelectKBest(f_regression, k = k)
    _ = featSelector.fit(dataFrame.drop(target, axis = 1), dataFrame[target])
    
    featScores = pd.DataFrame()
    featScores['F score'] = featSelector.scores_
    featScores['P value'] = featSelector.pvalues_
    featScores['Support'] = featSelector.get_support()
    featScores['Attribute'] = dataFrame.drop(target, axis = 1).columns
    
    return featScores

def indicesOfTopK(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(base.BaseEstimator, base.TransformerMixin):   
    
    def __init__(self, featureImportance, k):
        self.featureImportance = featureImportance
        self.k = k
    
    def fit(self, X, y = None):
        self.featureIndices_ = indicesOfTopK(self.featureImportance, self.k)
        return self
    
    def transform(self, X):
        return X[:, self.featureIndices_]
    



df = pd.DataFrame(exploreTrain.X_, columns = exploreTrain.X_.columns.tolist())
df['label'] = exploreTrain.y_
df['label'].fillna(0, inplace = True)
k = 10
featScores = selectKBestRef(df, 'label', k = k)

featScoresSorted = featScores.sort_values(['F score', 'P value'], ascending = [False, False])
print('Feature Score for a linear regression using correlation \n')
print(featScoresSorted[featScoresSorted['Support'] == True][:k])

featureFScores = featScores[['F score']].values.flatten()

sortedNames = np.array(featScoresSorted)[:k, 3]
sortedImportances = np.array(featScoresSorted)[:k, 0]

p = qp.plotter.QuickPlot(fig = plt.figure(), chartProp = 10, plotOrientation = 'square')
ax = p.makeCanvas(title = '', xLabel = '', yLabel = ''
                  ,yShift = 0.8, position = 111)
p.qpBarH(y = sortedNames
         ,counts = sortedImportances
         ,ax = ax
         )
ax.tick_params(axis = 'both', labelsize = 10)
