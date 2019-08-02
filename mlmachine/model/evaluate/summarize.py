import ast
import csv
import sys
import time
from timeit import default_timer as timer

import time

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import pandas as pd

import sklearn.model_selection as model_selection

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

from prettierplot.plotter import PrettierPlot
from prettierplot import style


def topBayesOptimModels(self, resultsRaw, numModels):
    models = {}
    for estimator in resultsRaw["estimator"].unique():
        estDf = resultsRaw[resultsRaw["estimator"] == estimator].sort_values(
            ["mean"], ascending=[False]
        )["iteration"][:numModels]
        models[estimator] = estDf.values.tolist()
    return models

