
import numpy as np
import pandas as pd
import sklearn.preprocessing as prepocessing
import sklearn.base as base
import sklearn.impute as impute


class CategoricalImputer(base.TransformerMixin):
    """
    Info:
        Description:
            Impute nominal categorical columns with mode value. Imputes
            training data features, and stores mode values to be used on
            validation and unseen data.
        Parameters:
            cols : list
                List of features to be imputer
            train : boolean, default = True
                Tells class whether we are imputing training data or unseen
                data.
            trainDict : dict, default = None
                Dictionary containing feature : mode pairs to be used
                to transform validation data. Only used when train = False.
                Retrieved from training data pipeline using named steps.
                Variable to be retrieved is called colValueDict_.
    """    
    def __init__(self, cols, train = True, trainDict = None):
        
        self.cols = cols
        self.train = train
        self.trainDict = trainDict
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # Encode training data
        if self.train:
            self.colValueDict_ = {}
            for col in self.cols:
                X[col] = X[col].fillna(X[col].value_counts().index[0])

                # build colValueDict
                self.colValueDict_[col] = X[col].value_counts().index[0]

        # For each columns, fill nulls with most frequently occuring value in training data
        else:
            for col in self.cols:
                X[col] = X[col].fillna(self.trainDict[col])
        return X


        
class NumericalImputer(base.TransformerMixin):
    """
    Info:
        Description:
            Impute nominal numerical columns with certain value, as specified
            by the strategy parameter. Imputes training data features, and stores 
            impute values to be used on validation and unseen data.
        Parameters:
            cols : list
                List of features to be imputer
            strategy : string, default = 'mean
                Imputing stategy. Takes values 'mean', 'median' and 'most_frequent'
            train : boolean, default = True
                Tells class whether we are imputing training data or unseen
                data.
            trainDict : dict, default = None
                Dictionary containing feature : value pairs to be used
                to transform validation data. Only used when train = False.
                Retrieved from training data pipeline using named steps.
                Variable to be retrieved is called colValueDict_.
    """    
    def __init__(self, cols, strategy = 'mean', train = True, trainDict = None):
        self.cols = cols
        self.strategy = strategy
        self.train = train
        self.trainDict = trainDict
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # Encode training data
        if self.train:
            self.colValueDict_ = {}
            for col in self.cols:
                imputed = impute.SimpleImputer(missing_values = np.nan, strategy = self.strategy)
                X[col] = imputed.fit_transform(X[[col]])
                
                # build colValueDict
                self.colValueDict_[col] = imputed.statistics_

        # For each columns, fill nulls with most frequently occuring value in training data
        else:
            for col in self.cols:
                X[col] = X[col].fillna(self.trainDict[col])
        return X

class ConstantImputer(base.TransformerMixin):
    """
    Info:
        Description:
            Impute specified columns with a specific value. Intended to
            be used when a missing value conveys that an observation does not
            have that attribute
        Parameters:
            cols : list
                List of features to be imputer
            fill : str, default = 'Absent'
                Default fill value
    """    
    def __init__(self, cols, fill = 'Absent'):
        self.cols = cols
        self.fill = fill
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        for col in self.cols:
            imputed = impute.SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = self.fill)
            X[col] = imputed.fit_transform(X[[col]])            
        return X


class ContextImputer(base.TransformerMixin):
    """
    Info:
        Description:
            Impute nominal numerical columns with certain value, as specified
            by the strategy parameter. Imputes training data features, and stores 
            impute values to be used on validation and unseen data.
        Parameters:
            cols : list
                List of features to be imputer
            strategy : string, default = 'mean'
                Imputing stategy. Takes values 'mean', 'median' and 'most_frequent'
            train : boolean, default = True
                Tells class whether we are imputing training data or unseen
                data.
            trainDf : dict, default = None
                Dataframe containing values to be mapped to replace nulls
                in validation data. Only used when train = False. Retrieved 
                from training data pipeline using named steps. Variable to 
                be retrieved is called fillDf.
    """    
    def __init__(self, nullCol, contextCol, strategy = 'mean', train = True, trainDf = None):
        self.nullCol = nullCol
        self.contextCol = contextCol
        self.strategy = strategy
        self.train = train
        self.trainDf = trainDf
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # Encode training data
        if self.train:
            if self.strategy == 'mean':
                self.fillDf = X.groupby(self.contextCol).mean()[self.nullCol]
            elif self.strategy == 'median':
                self.fillDf = X.groupby(self.contextCol).median()[self.nullCol]
            elif self.strategy == 'most_frequent':
                self.fillDf = X.groupby(self.contextCol)[self.nullCol].agg(lambda x: x.value_counts().index[0])

            self.fillDf = self.fillDf.reset_index()
            
            X[self.nullCol] = np.where(X[self.nullCol].isnull(), X[self.contextCol].map(self.fillDf.set_index(self.contextCol)[self.nullCol]), X[self.nullCol])

        # For each columns, fill nulls with most frequently occuring value in training data
        else:
            X[self.nullCol] = np.where(X[self.nullCol].isnull(), X[self.contextCol].map(self.trainDf.set_index(self.contextCol)[self.nullCol]), X[self.nullCol])
        return X
