
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from statsmodels.stats.weightstats import ztest
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats
from scipy import special

from IPython.display import display_html

import os
import sys

from prettierplot.plotter import PrettierPlot
from prettierplot import style

def edaTransformInitial(self, data, name):
    """

    """    
    p = PrettierPlot(chartProp = 15)
    
    # Before, distribution / kernel density plot
    ax = p.makeCanvas(title = 'Dist/KDE - {} (Initial)'.format(name), xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 221)
    p.prettyDistPlot(data, color = style.styleHexMid[0], fit = stats.norm, xRotate = True, ax = ax)
    plt.xticks([]); plt.yticks([])

    # Before, QQ plot
    ax = p.makeCanvas(title = 'Probability plot - {} (Initial)'.format(name), xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 222)
    p.prettyProbPlot(data, plot = ax)
    plt.xticks([]); plt.yticks([])
    
    # spacing
    # plt.subplots_adjust(wspace = 0.1)

def edaTransformLog1(self, data, name):
    """

    """    
    p = PrettierPlot(chartProp = 15)
    
    ax = p.makeCanvas(title = 'Dist/KDE - {} (log+1)'.format(name), xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 223)
    p.prettyDistPlot(np.log1p(data), color = style.styleHexMid[0], fit = stats.norm, xRotate = True, ax = ax)
    plt.xticks([]); plt.yticks([])

    # After, QQ plot
    ax = p.makeCanvas(title = 'Probability plot - {} (log+1)'.format(name), xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 224)
    p.prettyProbPlot(np.log1p(data), plot = ax)
    plt.xticks([]); plt.yticks([])

    # spacing
    # plt.subplots_adjust(wspace = 0.1, hspace = 0.3)

def edaTransformBoxCox(self, data, name, lmbda):
    """

    """    
    p = PrettierPlot(chartProp = 15)
    
    ax = p.makeCanvas(title = 'Dist/KDE - {} (BoxCox, {})'.format(name,lmbda), xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 223)
    p.prettyDistPlot(special.boxcox1p(data, lmbda), color = style.styleHexMid[0], fit = stats.norm, xRotate = True, ax = ax)
    plt.xticks([]); plt.yticks([])

    # After, QQ plot
    ax = p.makeCanvas(title = 'Probability plot - {} (BoxCox, {})'.format(name,lmbda), xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 224)
    p.prettyProbPlot(special.boxcox1p(data, lmbda), plot = ax)
    plt.xticks([]); plt.yticks([])

    # spacing
    # plt.subplots_adjust(wspace = 0.1, hspace = 0.3)