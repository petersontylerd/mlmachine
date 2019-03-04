
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from statsmodels.stats.weightstats import ztest
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

from IPython.display import display_html

import os
import sys

sys.path.append('/main')
from quickplot.plotter import QuickPlot
from quickplot import style

def edaMissingSummary(self):
    
    
    totalMissing = self.X_.isnull().sum()
    percentMissing = self.X_.isnull().sum() / len(self.X_) * 100
    percentMissing = pd.DataFrame({'Total missing' : totalMissing, 'Percent missing' : percentMissing})
    percentMissing = percentMissing[~(percentMissing['Percent missing'].isna()) 
                                    & (percentMissing['Percent missing'] > 0)].sort_values(['Percent missing'] , ascending = False)

    display(percentMissing)
    
    p = QuickPlot(fig = plt.figure(), chartProp = 15)
    ax = p.makeCanvas(title = 'Percent missing by feature', xLabel = '', yLabel = ''
                    ,yShift = 0.8, position = 221)
    p.qpBarV(x = percentMissing.index
            ,counts = percentMissing['Percent missing']
            ,labelRotate = 90 #if len(uniqueVals) >= 4 else 0
            ,color = style.styleHexMid[2]
            ,yUnits = 'p'
            ,ax = ax)                        
        
