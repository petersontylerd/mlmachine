
import ast
import csv
import sys
import time
from timeit import default_timer as timer
global ITERATION
import time

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from hyperopt.pyll.stochastic import sample

import sklearn.model_selection as model_selection

# Modeling extensions
import sklearn.decomposition as decomposition
import sklearn.discriminant_analysis as discriminant_analysis
import sklearn.ensemble as ensemble
import sklearn.gaussian_process as gaussian_process
import sklearn.linear_model as linear_model
import sklearn.kernel_ridge as kernel_ridge
import sklearn.naive_bayes as naive_bayes
import sklearn.neighbors as neighbors
import sklearn.svm as svm
import sklearn.tree as tree

import xgboost
import lightgbm
import catboost

# sys.path.append('/main')
from prettierplot.plotter import PrettierPlot
from prettierplot import style

# set optimization parameters
def objective(space, resultsDir = None, model = '', X = None, y = None, scoring = None
                ,n_folds = None, n_jobs = None, verbose = None):
    """
    Documentation:
        Description:
            desc
        Parameters:
            desc : asdf, default = 
                desc
        Returns:
            desc
    """
    global ITERATION
    ITERATION = 0
    ITERATION += 1
    start = timer()
    
    # convert select float params to int
    for param in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        if param in space.keys():
            space[param] = int(space[param])
    
    cv = model_selection.cross_val_score(estimator = eval('{0}(**{1})'.format(model, space))
                                         ,X = X
                                         ,y = y
                                         ,verbose = verbose
                                         ,n_jobs = n_jobs
                                         ,cv = n_folds
                                         ,scoring = 'neg_mean_squared_error' if scoring == 'rmsle' else scoring)
    run_time = timer() - start
    
    # Extract the best score
    if scoring == 'accuracy':
        loss = 1 - cv.mean()
    elif scoring == 'neg_mean_squared_error':
        loss = abs(cv.mean())
    elif scoring == 'rmsle':
        cv = np.sqrt(abs(cv))
        loss = np.mean(cv)
    
    # export results to CSV
    outFile = resultsDir
    with open(outFile, 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([ITERATION, model, space, loss, cv.min(), cv.mean(), cv.max(), cv.std(), run_time, STATUS_OK]) 

    return {'iteration' : ITERATION
            ,'estimator' : model
            ,'params' : space
            ,'loss' : loss                   # required
            ,'min' : cv.min()
            ,'mean' : cv.mean()
            ,'max' : cv.max()
            ,'std' : cv.std()
            ,'train_time' : run_time
            ,'status' : STATUS_OK}          # required

def execBayesOptimSearch(self, allSpace, resultsDir, X, y, scoring, n_folds, n_jobs, iters, verbose):
    """

    """
    with open(resultsDir, 'w', newline = '') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['iteration','estimator','params','loss','min','mean','max','std','train_time','status'])

    for estimator in allSpace.keys():
        space = allSpace[estimator]
        print(estimator)
        # reset objective function defaults
        objective.__defaults__ = (resultsDir, estimator, X, y, scoring, n_folds, n_jobs, verbose)
        
        # Run optimization
        print('#'*100)
        print('\nTuning {0}\n'.format(estimator))
        best = fmin(fn = objective
                    ,space = space
                    ,algo = tpe.suggest
                    ,max_evals = iters
                    ,trials = Trials()
                    ,verbose = 1
                )

def unpackParams(self, results):
    clfs = results['estimator'].unique()
    resultsDf = {}
    for clf in clfs:
        resultsClf = results[results['estimator'] == clf].reset_index(drop = True)
        
        # Create a new dataframe for storing parameters
        paramsClf = pd.DataFrame(columns = list(ast.literal_eval(resultsClf.loc[0, 'params']).keys()),
                                    index = list(range(len(resultsClf))))
        
        # Add the results with each parameter a different column
        for i, params in enumerate(resultsClf['params']):
            paramsClf.loc[i, :] = list(ast.literal_eval(params).values())

        # add columns for loss and iter number
        paramsClf['loss'] = resultsClf['loss']
        paramsClf['iteration'] = resultsClf['iteration']
        
        resultsDf[clf] = paramsClf
    return resultsDf

def lossPlot(self, resultsDf):
    for clf, df in resultsDf.items():
        lossDf = pd.DataFrame(np.stack((df['loss'], df['iteration'])
                                ,axis = -1)
                            ,columns = ['loss','iteration'])
        p = PrettierPlot(fig = plt.figure(), chartProp = 8, plotOrientation = 'square')
        ax = p.makeCanvas(title = 'Regression plot\n* {}'.format(clf), yShift = 0.8, position = 111)
        p.qpRegPlot(x = 'iteration'
                    ,y = 'loss'
                    ,data = lossDf
                    ,yUnits = 'fff'
                    ,ax = ax)

def samplePlot(self, sampleSpace, nIter):
    # iterate through each parameter 
    for param in sampleSpace.keys():
        # create data to represent theoretical distribution
        theoreticalDist = []
        for _ in range(nIter):
            theoreticalDist.append(sample(sampleSpace)[param])
        theoreticalDist = np.array(theoreticalDist)

        p = PrettierPlot(fig = plt.figure(), chartProp = 8, plotOrientation = 'square')
        ax = p.makeCanvas(title = 'Actual vs. theoretical plot\n* {}'.format(param), yShift = 0.8, position = 111)
        p.qpKde(theoreticalDist
                ,color = style.styleHexMid[0]
                ,yUnits = 'p'
                ,xUnits = 'fff' if np.max(theoreticalDist) <= 5.0 else 'ff'
                ,ax = ax)

def paramPlot(self, results, allSpace, nIter):
    """

    """
    # iterate through model name/df result key/value pairs
    for clf, df in results.items():
        # return space belonging to clf
        clfSpace = allSpace[clf]
        
        # iterate through each parameter 
        for param in clfSpace.keys():
            # create data to represent theoretical distribution
            theoreticalDist = []
            for _ in range(nIter):
                theoreticalDist.append(sample(clfSpace)[param])
            
            # clean up 
            theoreticalDist = ['None' if v is None else v for v in theoreticalDist]
            theoreticalDist = np.array(theoreticalDist)
            
            # clean up
            actualDist = df[param].tolist()
            actualDist = ['None' if v is None else v for v in actualDist]
            actualDist = np.array(actualDist)
            
            actualIterDf = pd.DataFrame(np.stack((df['iteration'], df[param])
                                ,axis = -1)
                            ,columns = ['iteration', param])
            
            # plot distributions for categorical params
            if any(isinstance(d, str) for d in theoreticalDist):
                
                p = PrettierPlot(fig = plt.figure(), chartProp = 8, plotOrientation = 'square')
                # theoretical plot
                uniqueVals, uniqueCounts = np.unique(theoreticalDist, return_counts = True)
                ax = p.makeCanvas(title = 'Theoretical plot\n* {}'.format(param), yShift = 0.8, position = 121)
                p.qpBarV(x = uniqueVals
                    ,counts = uniqueCounts
                    ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                    ,color = style.styleHexMid[2]
                    ,yUnits = 'f'
                    ,ax = ax)
                
                # actual plot
                uniqueVals, uniqueCounts = np.unique(actualDist, return_counts = True)
                ax = p.makeCanvas(title = 'Actual plot\n* {}'.format(param), yShift = 0.8, position = 122)
                p.qpBarV(x = uniqueVals
                    ,counts = uniqueCounts
                    ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                    ,color = style.styleHexMid[2]
                    ,yUnits = 'f'
                    ,ax = ax)
                
                plt.show()
            
            # plot distributions for continuous params
            else:
                # using dictionary to convert specific columns 
                convert_dict = {'iteration' : int, 
                                param : float
                               } 

                actualIterDf = actualIterDf.astype(convert_dict) 
                
                p = PrettierPlot(fig = plt.figure(), chartProp = 8, plotOrientation = 'square')
                ax = p.makeCanvas(title = 'Actual vs. theoretical plot\n* {}'.format(param), yShift = 0.8, position = 121)
                p.qpKde(theoreticalDist
                        ,color = style.styleHexMid[0]
                        ,yUnits = 'p'
                        ,xUnits = 'fff' if np.max(theoreticalDist) <= 5.0 else 'ff'
                        ,ax = ax)
                p.qpKde(actualDist
                        ,color = style.styleHexMid[1]
                        ,yUnits = 'p'
                        ,xUnits = 'fff' if np.max(actualDist) <= 5.0 else 'ff'
                        ,ax = ax)
                ax = p.makeCanvas(title = 'Regression plot\n* {}'.format(clf), yShift = 0.8, position = 122)
                p.qpRegPlot(x = 'iteration'
                            ,y = param
                            ,data = actualIterDf
                            ,yUnits = 'fff'
                            ,ax = ax)
                plt.show()
        
