import numpy as np
import pandas as pd

import sklearn.base as base

from collections import Counter



class OutlierIQR(base.TransformerMixin, base.BaseEstimator):
    """
    Info:
        Description: 
            a
    """
    def __init__(self, outlierCount, iqrStep, features, dropOutliers = False):
        self.outlierCount = outlierCount
        self.iqrStep = iqrStep
        self.features = features
        self.dropOutliers = dropOutliers
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        
        outlier_indices = []
        
        # iterate over features(columns)
        for col in self.features:
            Q1 = np.percentile(X[col], 25)
            Q3 = np.percentile(X[col], 75)
            IQR = Q3 - Q1
            
            # outlier step
            outlier_step = self.iqrStep * IQR
            
            # Determine a list of indices of outliers for feature col
            outlier_list_col = X[(X[col] < Q1 - outlier_step) | (X[col] > Q3 + outlier_step )].index
            
            # append the found outlier indices for col to the list of outlier indices 
            outlier_indices.extend(outlier_list_col)
            
        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)        
        self.outliers_ = list( k for k, v in outlier_indices.items() if v > self.outlierCount)
        
        if self.dropOutliers:
            X = X.drop(self.outliers_, axis = 0).reset_index(drop = True)

        return X  
