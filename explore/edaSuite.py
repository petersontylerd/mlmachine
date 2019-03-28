
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

def edaCatTargetCatFeat(self, skipCols = None):
    """
    Info:
        Description:
            a
        Parameters
            skipCols : list, default = None
            Column to skip over in visualization creation loop.
    """
    # Iterate through each feature within a feature type
    for feature in self.featureByDtype_['categorical']:
        if feature not in skipCols: 
            # Univariate summary
            uniSummDf = pd.DataFrame(columns = [feature, 'Count', 'Proportion'])
            uniqueVals, uniqueCounts = np.unique(self.X_[self.X_[feature].notnull()][feature], return_counts = True)
            for i, j in zip(uniqueVals, uniqueCounts):
                uniSummDf = uniSummDf.append({feature : i
                                        ,'Count' : j
                                        ,'Proportion' : j / np.sum(uniqueCounts) * 100
                                        }
                                    ,ignore_index = True)
            uniSummDf = uniSummDf.sort_values(by = ['Proportion'], ascending = False)
            
            # Bivariate summary
            biDf = pd.DataFrame(np.stack((self.X_[feature].values, self.y_)
                                    ,axis = -1)
                                ,columns = [feature, self.target[0]])
            biSummDf = biDf.groupby([feature] + self.target).size().reset_index()\
                                .pivot(columns = self.target[0], index = feature, values = 0)
            
            multiIndex = biSummDf.columns
            singleIndex = pd.Index([i for i in multiIndex.tolist()])
            biSummDf.columns = singleIndex
            biSummDf.reset_index(inplace = True)

            # Execute z-test
            if len(np.unique(biDf[biDf[feature].notnull()][feature])) == 2:
                
                # Total observations
                totalObs1 = biDf[(biDf[feature] == np.unique(biDf[feature])[0])][feature].shape[0]
                totalObs2 = biDf[(biDf[feature] == np.unique(biDf[feature])[1])][feature].shape[0]
                
                # Total positive observations
                posObs1 = biDf[(biDf[feature] == np.unique(biDf[feature])[0]) & (biDf[self.target[0]] == 1)][feature].shape[0]
                posObs2 = biDf[(biDf[feature] == np.unique(biDf[feature])[1]) & (biDf[self.target[0]] == 1)][feature].shape[0]
                
                z, pVal = proportions_ztest(count = (posObs1, posObs2), nobs = (totalObs1, totalObs2))
                
                statTestDf = pd.DataFrame(data = [{'z-test statistic' : z, 'p-value' : pVal}]
                                            ,columns = ['z-test statistic', 'p-value']
                                            ,index = [feature]).round(4)
                
                # Display summary tables
                self.dfSideBySide(dfs = (uniSummDf, biSummDf, statTestDf), names = ['Univariate summary', 'Biivariate summary', 'Statistical test'])
            
            else:            
                # Display summary tables
                self.dfSideBySide(dfs = (uniSummDf, biSummDf), names = ['Univariate summary', 'Biivariate summary'])
            
            # Instantiate charting object
            p = QuickPlot(fig = plt.figure(), chartProp = 15, plotOrientation = 'wide')
                    
            # Univariate plot
            ax = p.makeCanvas(title = 'Univariate\n* {}'.format(feature), yShift = 0.8, position = 121)
            
            p.qpBarV(x = uniqueVals
                    ,counts = uniqueCounts
                    ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                    ,color = style.styleHexMid[2]
                    ,yUnits = 'f'
                    ,ax = ax)                        
            
            # Bivariate plot
            ax = p.makeCanvas(title = 'Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 122)
            p.qpFacetCat(df = biSummDf
                        ,feature = feature
                        ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                        ,ax = ax)
            plt.show()
        # else:
        #     pass

def edaCatTargetNumFeat(self):
    """
    Info:
        Description:
    """
    # Iterate through each feature within a feature type
    for feature in self.featureByDtype_['continuous']:
        
        # Bivariate roll-up table
        biDf = pd.DataFrame(np.stack((self.X_[feature].values, self.y_)
                                ,axis = -1)
                            ,columns = [feature, self.target[0]])
        
        # Bivariate summary statistics
        biSummStatsDf = pd.DataFrame(columns = [feature, 'Count', 'Proportion', 'Mean', 'StdDv'])
        
        for labl in np.unique(self.y_):
            featureSlice = biDf[biDf[self.target[0]] == labl][feature]
        
            biSummStatsDf = biSummStatsDf.append({feature : labl
                                                    ,'Count' : len(featureSlice)
                                                    ,'Proportion' : len(featureSlice) / len(biDf[feature]) * 100
                                                    ,'Mean' : np.mean(featureSlice)
                                                    ,'StdDv' : np.std(featureSlice)
                                                    }
                                                ,ignore_index = True)
        
        # Display summary tables
        describeDf = pd.DataFrame(biDf[feature].describe()).reset_index()
        describeDf = describeDf.append({'index' : 'skew'
                                        ,feature : stats.skew(biDf[feature].values, nan_policy = 'omit')
                                        }
                                ,ignore_index = True)
        describeDf = describeDf.append({'index' : 'kurtosis'
                                        ,feature : stats.kurtosis(biDf[feature].values, nan_policy = 'omit')
                                        }
                                ,ignore_index = True)
        describeDf = describeDf.rename(columns = {'index' : ''})

        # Execute z-test or t-test
        if len(np.unique(self.y_)) == 2:
            s1 = biDf[biDf[self.target[0]] == biDf[self.target[0]].unique()[0]][feature]
            s2 = biDf[biDf[self.target[0]] == biDf[self.target[0]].unique()[1]][feature]
            if len(s1) > 30 and len(s2) > 30:
                z, pVal = ztest(s1, s2)
                
                statTestDf = pd.DataFrame(data = [{'z-test statistic' : z, 'p-value' : pVal}]
                                            ,columns = ['z-test statistic', 'p-value']
                                            ,index = [feature]).round(4)
            else:
                t, pVal = stats.ttest_ind(s1, s2)
                
                statTestDf = pd.DataFrame(data = [{'t-test statistic' : t, 'p-value' : pVal}]
                                            ,columns = ['t-test statistic', 'p-value']
                                            ,index = [feature]).round(4)
            self.dfSideBySide(dfs = (describeDf, biSummStatsDf, statTestDf)
                                ,names = ['Univariate stats', 'Bivariate stats', 'Statistical test'])
        else:
            self.dfSideBySide(dfs = (describeDf, biSummStatsDf)
                                ,names = ['Descriptive stats', 'Bivariate stats'])
            
        # Instantiate charting object
        p = QuickPlot(fig = plt.figure(), chartProp = 15, plotOrientation = 'wide')

        # Univariate plot
        ax = p.makeCanvas(title = 'Dist/KDE - Univariate\n* {}'.format(feature), yShift = 0.8, position = 151)
        p.qpDist(biDf[(biDf[feature].notnull())][feature].values
                ,color = style.styleHexMid[2]
                ,yUnits = 'ffff'
                ,ax = ax)
        
        # Probability plot
        ax = p.makeCanvas(title = 'Probability plot\n* {}'.format(feature), yShift = 0.8, position = 152)
        p.qpProbPlot(x = biDf[(biDf[feature].notnull())][feature].values
                    ,plot = ax)
        
        # Bivariate kernel density plot
        ax = p.makeCanvas(title = 'KDE - Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 153)
        for ix, labl in enumerate(np.unique(biDf[(biDf[feature].notnull())][self.target[0]].values)):
            p.qpKde(biDf[(biDf[feature].notnull()) & (biDf[self.target[0]] == labl)][feature].values
                    ,color = style.styleHexMid[ix]
                    ,yUnits = 'ffff'
                    ,ax = ax)

        # Bivariate histogram
        ax = p.makeCanvas(title = 'Hist - Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 154)
        for ix, labl in enumerate(np.unique(biDf[(biDf[feature].notnull())][self.target[0]].values)):
            p.qpFacetNum(biDf[(biDf[feature].notnull()) & (biDf[self.target[0]] == labl)][feature].values
                        ,color = style.styleHexMid[ix]
                        ,label = labl
                        ,alpha = 0.4)

        # Boxplot histogram
        ax = p.makeCanvas(title = 'Boxplot - Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 155)
        p.qpBoxPlotH(x = feature
                    ,y = self.target[0]
                    ,data = biDf
                    ,ax = ax)
        plt.show()

def edaNumTargetNumFeat(self):
    """
    Info:
        Description:
    """
    # Iterate through each feature within a feature type
    for feature in self.featureByDtype_['continuous']:
        
        ### Summary tables
        # Define bivariate dataframe
        biDf = pd.DataFrame(np.stack((self.X_[feature].values, self.y_)
                                ,axis = -1)
                            ,columns = [feature, self.target[0]])
        biDf[self.target[0]] = biDf[self.target[0]].astype(float)

        # Define summary tables
        describeDf = pd.DataFrame(biDf[feature].describe()).reset_index()
        
        # Instantiate charting object
        p = QuickPlot(fig = plt.figure(), chartProp = 15, plotOrientation = 'wide')

        # If continuous variable has fewer than a set number of unique variables, represent variable
        # as a categorical variable vs. a continuous target variable
        if len(np.unique(biDf[feature].values)) <= 20:
            
            describeDf = describeDf.rename(columns = {'index' : ''})                        
            
            # Bivariate summary statistics
            biSummStatsDf = pd.DataFrame(columns = [feature, 'Count', 'Proportion', 'Mean', 'StdDv'])                            
            for featureVal in np.unique(biDf[feature].values):
                featureSlice = biDf[(biDf[feature] == featureVal) & (biDf[feature].notnull())][feature]
            
                biSummStatsDf = biSummStatsDf.append({feature : featureVal
                                                        ,'Count' : len(featureSlice)
                                                        ,'Proportion' : len(featureSlice) / len(biDf[feature]) * 100
                                                        ,'Mean' : np.mean(featureSlice)
                                                        ,'StdDv' : np.std(featureSlice)
                                                        }
                                                    ,ignore_index = True)
            
            # Display summary dataframes
            self.dfSideBySide(dfs = (describeDf, biSummStatsDf), names = ['Univariate stats', 'Bivariate stats'])

            # Univariate plot
            uniqueVals, uniqueCounts = np.unique(self.X_[self.X_[feature].notnull()][feature], return_counts = True)
            ax = p.makeCanvas(title = 'Univariate\n* {}'.format(feature), yShift = 0.8, position = 131)
            p.qpBarV(x = uniqueVals
                ,counts = uniqueCounts
                ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                ,color = style.styleHexMid[2]
                ,yUnits = 'f'
                ,ax = ax)                        

            # Regression plot
            ax = p.makeCanvas(title = 'Regression plot\n* {}'.format(feature), yShift = 0.8, position = 132)
            p.qpRegPlot(x = feature
                        ,y = self.target[0]
                        ,data = biDf[biDf[feature].notnull()]
                        ,x_jitter = .2
                        ,ax = ax)

            # Bivariate box plot
            ax = p.makeCanvas(title = 'Box plot - Faceted by\n* {}'.format(feature), yShift = 0.8, position = 133)
            p.qpBoxPlotV(x = feature
                        ,y = self.target[0]
                        ,data = biDf[biDf[feature].notnull()]
                        ,color = style.genCmap(len(uniqueVals), [style.styleHexMid[0], style.styleHexMid[1], style.styleHexMid[2]])
                        ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                        ,ax = ax)

        # If continuous variable has greater than a set number of unique variables, represent variable
        # as a continuous variable vs. a continuous target variable
        else:
            
            # Add skew and curtosis to describeDf
            describeDf = describeDf.append({'index' : 'skew', feature : stats.skew(biDf[feature].values, nan_policy = 'omit')
                                        }, ignore_index = True)
            describeDf = describeDf.append({'index' : 'kurtosis', feature : stats.kurtosis(biDf[feature].values, nan_policy = 'omit')
                                        }, ignore_index = True)
            describeDf = describeDf.rename(columns = {'index' : ''})
            
            # Display summary dataframes
            display(describeDf)

            # Univariate plot
            ax = p.makeCanvas(title = 'Dist/KDE - Univariate\n* {}'.format(feature), yShift = 0.8, position = 131)
            p.qpDist(biDf[(biDf[feature].notnull())][feature].values                                
                    ,color = style.styleHexMid[2]
                    ,yUnits = 'ffff'
                    ,fit = stats.norm
                    ,ax = ax)
        
            # Probability plot
            ax = p.makeCanvas(title = 'Probability plot\n* {}'.format(feature), yShift = 0.8, position = 132)
            p.qpProbPlot(x = biDf[(biDf[feature].notnull())][feature].values
                        ,plot = ax)
        
            # Regression plot
            ax = p.makeCanvas(title = 'Regression plot\n* {}'.format(feature), yShift = 0.8, position = 133)
            p.qpRegPlot(x = feature
                        ,y = self.target[0]
                        ,data = biDf[biDf[feature].notnull()]
                        ,x_jitter = .1
                        ,ax = ax)                            
        plt.show()

def edaNumTargetCatFeat(self):
    """
    Info:
        Description:
    """
    # Iterate through each feature within a feature type
    for feature in self.featureByDtype_['categorical']:
        
        ### Summary tables
        # Univariate summary
        uniSummDf = pd.DataFrame(columns = [feature, 'Count', 'Proportion'])
        uniqueVals, uniqueCounts = np.unique(self.X_[self.X_[feature].notnull()][feature], return_counts = True)
        for i, j in zip(uniqueVals, uniqueCounts):
            uniSummDf = uniSummDf.append({feature : i
                                    ,'Count' : j
                                    ,'Proportion' : j / np.sum(uniqueCounts) * 100
                                    }
                                ,ignore_index = True)
        uniSummDf = uniSummDf.sort_values(by = ['Proportion'], ascending = False)
        
        # Bivariate summary
        biDf = pd.DataFrame(np.stack((self.X_[feature].values, self.y_)
                                ,axis = -1)
                            ,columns = [feature, self.target[0]])
        biDf[self.target[0]] = biDf[self.target[0]].astype(float)
        statsDict = {'N' : len, 'Median' : np.nanmedian, 'Mean' : np.nanmean, 'StdDev' : np.nanstd}
        biSummPivDf = pd.pivot_table(biDf
                                    ,index = feature
                                    ,aggfunc = {self.target[0] : statsDict})
        multiIndex = biSummPivDf.columns
        singleIndex = pd.Index([i[1] for i in multiIndex.tolist()])
        biSummPivDf.columns = singleIndex
        biSummPivDf.reset_index(inplace = True)

        # Display summary tables
        self.dfSideBySide(dfs = (uniSummDf, biSummPivDf), names = ['Univariate summary', 'Bivariate summary'])
        
        ### Plots
        # Instantiate charting object
        p = QuickPlot(fig = plt.figure(), chartProp = 15, plotOrientation = 'wide')
        
        # Univariate plot
        ax = p.makeCanvas(title = 'Univariate\n* {}'.format(feature), yShift = 0.8, position = 121)
        
        # Select error catching block for resorting labels
        try:
            sorted(uniqueVals, key = int)
        except ValueError:
            pass
        else:
            # Sort uniqueVals/uniqueCounts for bar chart
            newIx = [sorted(list(uniqueVals), key = int).index(i) for i in list(uniqueVals)]
            uniqueVals = np.array(sorted(list(uniqueVals), key = int))
            uniqueCounts = np.array([y for x,y in sorted(zip(newIx, uniqueCounts))])
        
            # Sort temporary data frame for box plot
            biDf[feature] = biDf[feature].astype(int)

        p.qpBarV(x = uniqueVals
                ,counts = uniqueCounts
                ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                ,color = style.styleHexMid[2]
                ,yUnits = 'f'
                ,ax = ax)                 
                                
        # Bivariate box plot
        ax = p.makeCanvas(title = 'Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 122)
        p.qpBoxPlotV(x = feature
                    ,y = self.target[0]
                    ,data = biDf[biDf[feature].notnull()].sort_values([feature])
                    ,color = style.genCmap(len(uniqueVals), [style.styleHexMid[0], style.styleHexMid[1], style.styleHexMid[2]])
                    ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                    ,ax = ax)                        
        plt.show()

def dfSideBySide(self, dfs, names = []):
    """
    Info:
        Description:

        Parameters:
    """
    html_str = ''
    if names:
        html_str += ('<tr>' + 
                    ''.join(f'<td style="text-align:center">{name}</td>' for name in names) + 
                    '</tr>')
    html_str += ('<tr>' + 
                ''.join(f'<td style="vertical-align:top"> {df.to_html(index=False)}</td>' 
                        for df in dfs) + 
                '</tr>')
    html_str = f'<table>{html_str}</table>'
    html_str = html_str.replace('table','table style="display:inline"')
    display_html(html_str, raw=True)

