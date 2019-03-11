
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
    
    # add column describing percent of values that are zero
    skewness['PctZero'] = np.nan
    for col in self.featureByDtype_['continuous']:

        try:
            skewness.loc[col]['PctZero'] = self.X_[self.X_[col] == 0][col].value_counts() / len(self.X_)
        except ValueError:
            skewness.loc[col]['PctZero'] = 0.0
    skewness = skewness.sort_values(['PctZero'])
    display(skewness)
    
class SkewTransform(base.TransformerMixin, base.BaseEstimator):
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
            pctZeroMax : float
                Maximum percent zero values a column is allowed to have
                in order to be transformed. Must be a value between 0.0 and 1.0.
            contCols : list
                List of columns to be evaluated for transformation
            train : boolean, default = True
                Controls whether to fit_transform training data or
                transform validation data using lambdas determined
                from training data
            classDict : dict, default = None
                Dictionary containing feature : lambda pairs to be used
                to transform validation data using Box-Cox +1 transformation. 
                Only used when train = False. Variable to be retrieved is 
                called colValueDict_.
    """
    def __init__(self, cols, skewMin = None, pctZeroMax = None, train = True, trainDict = None):
        
        self.cols = cols
        self.skewMin = skewMin
        self.pctZeroMax = pctZeroMax
        self.train = train
        self.trainDict = trainDict
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # Encode training data
        if self.train:
            skews = X[self.cols].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending = False).to_dict()
            self.colValueDict_ = {}
            for col, skew in skews.items():
                # determine percent of column values that are zero
                try:
                    pctZero = (X[X[col] == 0][col].value_counts() / len(X)).values[0]

                except IndexError:
                    pctZero = 0.0
                
                # if the skew value is greater than the minimum skew allowed and pctZero is lower than a maximum allowed
                if skew >= self.skewMin and pctZero <= self.pctZeroMax:
                    
                    # build dictionary of lambda : skew pairs
                    lambdaDict = {}
                    for lmbda in np.linspace(-2.0, 2.0, 500):
                        lambdaDict[lmbda] = abs(stats.skew(special.boxcox1p(X[col], lmbda)))
                    
                    # detemine value of lambda that result in a skew closest to 0

                    lowLambda = min(lambdaDict.items(), key = lambda kv : abs(kv[1] - 0.0))[0]

                    X[col] = special.boxcox1p(X[col], lowLambda)
                    self.colValueDict_[col] = lowLambda
                    print('transformed {}'.format(col))
            
        # Encode validation data with training data encodings.
        else:
            for col, lmbda in self.trainDict.items():
                X[col] = special.boxcox1p(X[col], lmbda)
        return X

class Binner(base.TransformerMixin):
    """
    Info:
        Description:
            Bin continuous columns into specified segments. Bins training data 
            features, and stores the cut points to be used on validation and
            unseen data.
        Parameters:
            colBinLabelDict : dictionary
                Dictionary containing col : label pairs. Custom bin labels to be 
                used for each paired column. The bin size is calculated based off 
                of the label length. 
            train : boolean, default = True
                Tells class whether we are binning training data or unseen
                data.
            trainDict : dict, default = None
                Dictionary containing feature : mode pairs to be used
                to transform validation data. Only used when train = False.
                Retrieved from training data pipeline using named steps.
                Variable to be retrieved is called colValueDict_.
    """
    def __init__(self, colBinLabelDict, train = True, trainDict = None):
        self.colBinLabelDict = colBinLabelDict
        self.train = train
        self.trainDict = trainDict
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # Encode training data
        if self.train:
            self.colValueDict_ = {}
            for col, label in self.colBinLabelDict.items():
                # retrieve bin cutoffs from original column
                _, bins = pd.cut(X[col], bins = len(label), labels = label, retbins = True)
                
                # add binned version of original column to dataset 
                X['{}_{}'.format(col,'binned')] = pd.cut(X[col], bins = len(label), labels = False)    
                
                # build colValueDict
                self.colValueDict_[col] = bins
                
        # For each column, bin the values based on the cut-offs learned on training data
        else:
            for col, bins in self.trainDict.items():
                trainBins = pd.IntervalIndex.from_breaks(bins)
                X['{}_{}'.format(col,'binned')] = pd.cut(X[col], bins = trainBins).codes
        return X