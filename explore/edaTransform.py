
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

sys.path.append('/main')
from quickplot.plotter import QuickPlot
from quickplot import style

def edaTransformInitial(self, data, name):
    """

    """    
    p = QuickPlot(fig = plt.figure(), chartProp = 15)
    
    # Before, distribution / kernel density plot
    ax = p.makeCanvas(title = 'Dist/KDE - {} (Initial)'.format(name), xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 221)
    p.qpDist(data, color = style.styleHexMid[0], fit = stats.norm, xRotate = True, ax = ax)
    plt.xticks([]); plt.yticks([])

    # Before, QQ plot
    ax = p.makeCanvas(title = 'Probability plot - {} (Initial)'.format(name), xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 222)
    p.qpProbPlot(data, plot = ax)
    plt.xticks([]); plt.yticks([])
    
    # spacing
    # plt.subplots_adjust(wspace = 0.1)

def edaTransformLog1(self, data, name):
    """

    """    
    p = QuickPlot(fig = plt.figure(), chartProp = 15)
    
    ax = p.makeCanvas(title = 'Dist/KDE - {} (log+1)'.format(name), xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 223)
    p.qpDist(np.log1p(data), color = style.styleHexMid[0], fit = stats.norm, xRotate = True, ax = ax)
    plt.xticks([]); plt.yticks([])

    # After, QQ plot
    ax = p.makeCanvas(title = 'Probability plot - {} (log+1)'.format(name), xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 224)
    p.qpProbPlot(np.log1p(data), plot = ax)
    plt.xticks([]); plt.yticks([])

    # spacing
    # plt.subplots_adjust(wspace = 0.1, hspace = 0.3)

def edaTransformBoxCox(self, data, name, lmbda):
    """

    """    
    p = QuickPlot(fig = plt.figure(), chartProp = 15)
    
    ax = p.makeCanvas(title = 'Dist/KDE - {} (BoxCox, {})'.format(name,lmbda), xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 223)
    p.qpDist(special.boxcox1p(data, lmbda), color = style.styleHexMid[0], fit = stats.norm, xRotate = True, ax = ax)
    plt.xticks([]); plt.yticks([])

    # After, QQ plot
    ax = p.makeCanvas(title = 'Probability plot - {} (BoxCox, {})'.format(name,lmbda), xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 224)
    p.qpProbPlot(special.boxcox1p(data, lmbda), plot = ax)
    plt.xticks([]); plt.yticks([])

    # spacing
    # plt.subplots_adjust(wspace = 0.1, hspace = 0.3)