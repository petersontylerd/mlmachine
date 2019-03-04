
import numpy as np
import pandas as pd

from scipy import stats
from scipy import special

import sklearn.base as base


def skewSummary(self):
    """

    """
    skewness = self.X_[self.featureByDtype_['continuous']].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending = False)
    skewness = pd.DataFrame({'Skew' : skewness})
    display(skewness)
    
class SkewTransform(base.TransformerMixin, base.BaseEstimator):
    """
    
    """
    def __init__(self, skewMin, contCols, train = True, skewLambdas = None):
        """
        Info:
            Description:
                Performs Box-Cox +1 transformation on continuous features
                with an absolute skew value above skewMin. The lambda chosen
                for each feature is the value that gets the skew closest to
                zero. The same lambda values are reused on unseen data.
            Parameters:
                skewMin : float
                    Minimum absolute skew of feature needed in order 
                    for feature to be transformed
                contCols : list
                    List of columns to be evaluated for transformation
                train : boolean, default = True
                    Controls whether to fit_transform training data or
                    transform validation data using lambdas determined
                    from training data
                classDict : dict, default = None
                    Dictionary containing feature : lambda pairs to be used
                    to transform validation data using Box-Cox +1 transformation. 
                    Only used when train = False.
        """        
        self.skewMin = skewMin
        self.contCols = contCols
        self.train = train
        self.skewLambdas = skewLambdas

        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # Encode training data
        if self.train:
            skews = X[self.contCols].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending = False).to_dict()
            self.skewLambdas_ = {}
            for col, skew in skews.items():
                # if the skew value is greater than the minimum skew allowed
                if skew >= self.skewMin:
                    
                    # build dictionary of lambda : skew pairs
                    lambdaDict = {}
                    for lmbda in np.linspace(-2.0, 2.0, 500):
                        lambdaDict[lmbda] = abs(stats.skew(special.boxcox1p(X[col], lmbda)))
                    
                    # detemine value of lambda that result in a skew closest to 0

                    lowLambda = min(lambdaDict.items(), key = lambda kv : abs(kv[1] - 0.0))[0]

                    X[col] = special.boxcox1p(X[col], lowLambda)
                    self.skewLambdas_[col] = lowLambda
            
        # Encode validation data with training data encodings.
        else:
            for col, lmbda in self.skewLambdas.items():
                X[col] = special.boxcox1p(X[col], lmbda)
        return X

# def skewTransform(self, skewMin):
#     """

#     """
#     # Iterate through all feature names in skews_
#     for i, j in self.skews_['Skew'].items():
        
#         # if the skew value is greater than the minimum skew allowed
#         if j >= skewMin:
            
#             # build dictionary of lambda : skew pairs
#             lambdaDict = {}
#             for lmbda in np.linspace(-2.0, 2.0, 500):
#                 lambdaDict[lmbda] = abs(stats.skew(special.boxcox1p(self.X_[i], lmbda)))
            
#             # detemine value of lambda that result in a skew closest to 0

#             lowSkew = min(lambdaDict.items(), key = lambda kv : abs(kv[1] - 0.0))

#             self.X_[i] = special.boxcox1p(self.X_[i], lowSkew[0])
#             print('{}:\n Initial skew = {}, Post Box-Cox (lambda {}) = {}'.format(i, np.round(j,5), np.round(lowSkew[0],5), np.round(lowSkew[1],5)))
            


