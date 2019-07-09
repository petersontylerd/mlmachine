
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

from prettierplot.plotter import PrettierPlot
from prettierplot import style

def edaCatTargetCatFeat(self, skipCols = []):
    """
    Documentation:
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
            uniqueVals, uniqueCounts = np.unique(self.data[self.data[feature].notnull()][feature], return_counts = True)
            for i, j in zip(uniqueVals, uniqueCounts):
                uniSummDf = uniSummDf.append({feature : i
                                        ,'Count' : j
                                        ,'Proportion' : j / np.sum(uniqueCounts) * 100
                                        }
                                    ,ignore_index = True)
            uniSummDf = uniSummDf.sort_values(by = ['Proportion'], ascending = False)
            
            # Bivariate summary
            biDf = pd.concat([self.data[feature], self.target], axis = 1)

            biSummDf = biDf.groupby([feature] + [self.target.name]).size().reset_index()\
                           .pivot(columns = self.target.name, index = feature, values = 0)
            
            multiIndex = biSummDf.columns
            singleIndex = pd.Index([i for i in multiIndex.tolist()])
            biSummDf.columns = singleIndex
            biSummDf.reset_index(inplace = True)

            # add PercentPositive column
            if len(np.unique(self.target)):
                biSummDf['PercentPositive'] = (biSummDf[1] / (biSummDf[1] + biSummDf[0])) * 100

            # Execute z-test
            if len(np.unique(biDf[biDf[feature].notnull()][feature])) == 2:
                
                # Total observations
                totalObs1 = biDf[(biDf[feature] == np.unique(biDf[feature])[0])][feature].shape[0]
                totalObs2 = biDf[(biDf[feature] == np.unique(biDf[feature])[1])][feature].shape[0]
                
                # Total positive observations
                posObs1 = biDf[(biDf[feature] == np.unique(biDf[feature])[0]) & (biDf[self.target.name] == 1)][feature].shape[0]
                posObs2 = biDf[(biDf[feature] == np.unique(biDf[feature])[1]) & (biDf[self.target.name] == 1)][feature].shape[0]
                
                z, pVal = proportions_ztest(count = (posObs1, posObs2), nobs = (totalObs1, totalObs2))
                
                statTestDf = pd.DataFrame(data = [{'z-test statistic' : z, 'p-value' : pVal}]
                                            ,columns = ['z-test statistic', 'p-value']
                                            ,index = [feature]).round(4)
                
                # Display summary tables
                self.dfSideBySide(dfs = (uniSummDf, biSummDf, statTestDf), names = ['Univariate summary', 'Biivariate summary', 'Statistical test'])
                if 'PercentPositive' in biSummDf: biSummDf = biSummDf.drop(['PercentPositive'], axis = 1)
        
            else:            
                # Display summary tables
                self.dfSideBySide(dfs = (uniSummDf, biSummDf), names = ['Univariate summary', 'Biivariate summary'])
                if 'PercentPositive' in biSummDf: biSummDf = biSummDf.drop(['PercentPositive'], axis = 1)
            
            # set label rotation angle    
            lenUniqueVal = len(uniqueVals)
            avgLenUniqueVal = (sum(map(len, str(uniqueVals))) / len(uniqueVals))
            if lenUniqueVal <= 4 and avgLenUniqueVal <= 12:
                rotation = 0
            elif lenUniqueVal >= 5 and lenUniqueVal <= 8 and avgLenUniqueVal <= 8:
                rotation = 0
            elif lenUniqueVal >= 9 and lenUniqueVal <= 14 and avgLenUniqueVal <= 4:
                rotation = 0
            else:
                rotation = 90

            # Instantiate charting object
            p = PrettierPlot(chartProp = 15, plotOrientation = 'wide')
                    
            # Univariate plot
            ax = p.makeCanvas(title = 'Univariate\n* {}'.format(feature), position = 121)
            
            p.prettyBarV(x = list(map(str, uniqueVals.tolist()))
                        ,counts = uniqueCounts
                        ,labelRotate = rotation
                        ,color = style.styleHexMid[2]
                        ,yUnits = 'f'
                        ,ax = ax
                )
            
            # Bivariate plot
            ax = p.makeCanvas(title = 'Faceted by target\n* {}'.format(feature), position = 122)
            p.prettyFacetCat(df = biSummDf
                            ,feature = feature
                            ,labelRotate = rotation
                            ,ax = ax
                )
            plt.show()


def edaCatTargetNumFeat(self):
    """
    Documentation:
        Description:
    """
    # sns.set(rc = style.rcGrey)

    # Iterate through each feature within a feature type
    for feature in self.featureByDtype_['continuous']:
        
        # Bivariate roll-up table
        biDf = pd.concat([self.data[feature], self.target], axis = 1)
        
        # Bivariate summary statistics
        biSummStatsDf = pd.DataFrame(columns = [feature, 'Count', 'Proportion', 'Mean', 'StdDv'])
        
        for labl in np.unique(self.target):
            featureSlice = biDf[biDf[self.target.name] == labl][feature]
        
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
                                        ,feature : np.round(stats.skew(biDf[feature].values, nan_policy = 'omit'), 5)
                                        }
                                ,ignore_index = True)
        describeDf = describeDf.append({'index' : 'kurtosis'
                                        ,feature : stats.kurtosis(biDf[feature].values, nan_policy = 'omit')
                                        }
                                ,ignore_index = True)
        describeDf = describeDf.rename(columns = {'index' : ''})

        # Execute z-test or t-test
        if len(np.unique(self.target)) == 2:
            s1 = biDf[(biDf[self.target.name] == biDf[self.target.name].unique()[0]) & (biDf[feature].notnull())][feature]
            s2 = biDf[(biDf[self.target.name] == biDf[self.target.name].unique()[1]) & (biDf[feature].notnull())][feature]
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
        p = PrettierPlot(chartProp = 15, plotOrientation = 'wide')

        # Univariate plot
        ax = p.makeCanvas(title = 'Dist/KDE - Univariate\n* {}'.format(feature), position = 151)
        p.prettyDistPlot(biDf[(biDf[feature].notnull())][feature].values
                ,color = style.styleHexMid[2]
                ,yUnits = 'ffff'
                ,ax = ax)
        
        # Probability plot
        ax = p.makeCanvas(title = 'Probability plot\n* {}'.format(feature), position = 152)
        p.prettyProbPlot(x = biDf[(biDf[feature].notnull())][feature].values
                    ,plot = ax)
        
        # Bivariate kernel density plot
        ax = p.makeCanvas(title = 'KDE - Faceted by target\n* {}'.format(feature), position = 153)
        for ix, labl in enumerate(np.unique(biDf[(biDf[feature].notnull())][self.target.name].values)):
            p.prettyKdePlot(biDf[(biDf[feature].notnull()) & (biDf[self.target.name] == labl)][feature].values
                    ,color = style.styleHexMid[ix]
                    ,yUnits = 'ffff'
                    ,ax = ax)

        # Bivariate histogram
        ax = p.makeCanvas(title = 'Hist - Faceted by target\n* {}'.format(feature), position = 154)
        for ix, labl in enumerate(np.unique(biDf[(biDf[feature].notnull())][self.target.name].values)):
            p.prettyHist(biDf[(biDf[feature].notnull()) & (biDf[self.target.name] == labl)][feature].values
                        ,color = style.styleHexMid[ix]
                        ,label = labl
                        ,alpha = 0.4)

        # Boxplot histogram
        ax = p.makeCanvas(title = 'Boxplot - Faceted by target\n* {}'.format(feature), position = 155)
        p.prettyBoxPlotH(x = feature
                    ,y = self.target.name
                    ,data = biDf
                    ,ax = ax)
        plt.show()

def edaNumTargetNumFeat(self):
    """
    Documentation:
        Description:
    """
    # sns.set(rc = style.rcGrey)
    
    # Iterate through each feature within a feature type
    for feature in self.featureByDtype_['continuous']:
        
        ### Summary tables
        # Define bivariate dataframe
        biDf = pd.concat([self.data[feature], self.target], axis = 1)
        
        # biDf = pd.DataFrame(np.stack((self.data[feature].values, self.y_)
        #                         ,axis = -1)
        #                     ,columns = [feature, self.target.name])
        
        biDf[self.target.name] = biDf[self.target.name].astype(float)

        # Define summary tables
        describeDf = pd.DataFrame(biDf[feature].describe()).reset_index()
        
        # Instantiate charting object
        p = PrettierPlot(chartProp = 15, plotOrientation = 'wide')

        # If continuous variable has fewer than a set number of unique variables, represent variable
        # as a categorical variable vs. a continuous target variable
        if len(np.unique(biDf[feature].values)) <= 20:
            
            describeDf = describeDf.rename(columns = {'index' : ''})                        
            
            # Bivariate summary statistics
            biSummStatsDf = pd.DataFrame(columns = [feature, 'Count', 'Proportion', 'Mean', 'StdDv'])                            
            uniqueVals, uniqueCounts = np.unique(self.data[self.data[feature].notnull()][feature], return_counts = True)
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

            # set rotation angle
            lenUniqueVal = len(uniqueVals)
            avgLenUniqueVal = (sum(map(len, str(uniqueVals))) / len(uniqueVals))
            if lenUniqueVal <= 4 and avgLenUniqueVal <= 12:
                rotation = 0
            elif lenUniqueVal >= 5 and lenUniqueVal <= 8 and avgLenUniqueVal <= 8:
                rotation = 0
            elif lenUniqueVal >= 9 and lenUniqueVal <= 14 and avgLenUniqueVal <= 4:
                rotation = 0
            else:
                rotation = 90

            # Univariate plot
            ax = p.makeCanvas(title = 'Univariate\n* {}'.format(feature), position = 131)
            p.prettyBarV(x = list(map(str, uniqueVals.tolist()))
                ,counts = uniqueCounts
                ,labelRotate = rotation
                ,color = style.styleHexMid[2]
                ,yUnits = 'f'
                ,ax = ax)                        

            # Regression plot
            ax = p.makeCanvas(title = 'Regression plot\n* {}'.format(feature), position = 132)
            p.prettyRegPlot(x = feature
                        ,y = self.target.name
                        ,data = biDf[biDf[feature].notnull()]
                        ,x_jitter = .2
                        ,ax = ax)
            
            # hide every other label if total number of levels is less than 5
            if lenUniqueVal <= 4:
                xmin, xmax = ax.get_xlim()
                ax.set_xticks(np.round(np.linspace(xmin, xmax, lenUniqueVal), 2))
                
            # Bivariate box plot
            ax = p.makeCanvas(title = 'Box plot - Faceted by\n* {}'.format(feature), position = 133)
            p.prettyBoxPlotV(x = feature
                        ,y = self.target.name
                        ,data = biDf[biDf[feature].notnull()]
                        ,color = style.genCmap(len(uniqueVals), [style.styleHexMid[0], style.styleHexMid[1], style.styleHexMid[2]])
                        ,labelRotate = rotation
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
            ax = p.makeCanvas(title = 'Dist/KDE - Univariate\n* {}'.format(feature), position = 131)
            p.prettyDistPlot(biDf[(biDf[feature].notnull())][feature].values                                
                    ,color = style.styleHexMid[2]
                    ,yUnits = 'ffff'
                    ,fit = stats.norm
                    ,xRotate = 45
                    ,ax = ax)
        
            # Probability plot
            ax = p.makeCanvas(title = 'Probability plot\n* {}'.format(feature), position = 132)
            p.prettyProbPlot(x = biDf[(biDf[feature].notnull())][feature].values
                        ,plot = ax)
        
            # Regression plot
            ax = p.makeCanvas(title = 'Regression plot\n* {}'.format(feature), position = 133)
            p.prettyRegPlot(x = feature
                        ,y = self.target.name
                        ,data = biDf[biDf[feature].notnull()]
                        ,x_jitter = .1
                        ,xRotate = 45
                        ,ax = ax)                            
        plt.show()

def edaNumTargetCatFeat(self):
    """
    Documentation:
        Description:
    """
    # sns.set(rc = style.rcGrey)

    # Iterate through each feature within a feature type
    for feature in self.featureByDtype_['categorical']:
        
        ### Summary tables
        # Univariate summary
        uniSummDf = pd.DataFrame(columns = [feature, 'Count', 'Proportion'])
        uniqueVals, uniqueCounts = np.unique(self.data[self.data[feature].notnull()][feature], return_counts = True)
        for i, j in zip(uniqueVals, uniqueCounts):
            uniSummDf = uniSummDf.append({feature : i
                                    ,'Count' : j
                                    ,'Proportion' : j / np.sum(uniqueCounts) * 100
                                    }
                                ,ignore_index = True)
        uniSummDf = uniSummDf.sort_values(by = ['Proportion'], ascending = False)
        
        # Bivariate summary
        biDf = pd.concat([self.data[feature], self.target], axis = 1)
        
        # biDf = pd.DataFrame(np.stack((self.data[feature].values, self.y_)
        #                         ,axis = -1)
        #                     ,columns = [feature, self.target.name])
        
        biDf[self.target.name] = biDf[self.target.name].astype(float)
        statsDict = {'N' : len, 'Median' : np.nanmedian, 'Mean' : np.nanmean, 'StdDev' : np.nanstd}
        biSummPivDf = pd.pivot_table(biDf
                                    ,index = feature
                                    ,aggfunc = {self.target.name : statsDict})
        multiIndex = biSummPivDf.columns
        singleIndex = pd.Index([i[1] for i in multiIndex.tolist()])
        biSummPivDf.columns = singleIndex
        biSummPivDf.reset_index(inplace = True)

        # Display summary tables
        self.dfSideBySide(dfs = (uniSummDf, biSummPivDf), names = ['Univariate summary', 'Bivariate summary'])
        
        ### Plots
        # Instantiate charting object
        p = PrettierPlot(chartProp = 15, plotOrientation = 'wide')
        
        # Univariate plot
        ax = p.makeCanvas(title = 'Univariate\n* {}'.format(feature), position = 121)
        
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

        # set rotation angle
        lenUniqueVal = len(uniqueVals)
        avgLenUniqueVal = (sum(map(len, str(uniqueVals))) / len(uniqueVals))
        if lenUniqueVal <= 4 and avgLenUniqueVal <= 12:
            rotation = 0
        elif lenUniqueVal >= 5 and lenUniqueVal <= 8 and avgLenUniqueVal <= 10:
            rotation = 0
        elif lenUniqueVal >= 9 and lenUniqueVal <= 14 and avgLenUniqueVal <= 6:
            rotation = 0
        else:
            rotation = 90

        p.prettyBarV(x = list(map(str, uniqueVals.tolist()))
                    ,counts = uniqueCounts
                    ,labelRotate = rotation
                    ,color = style.styleHexMid[2]
                    ,yUnits = 'f'
                    ,ax = ax
            )

        # hide every other label if total number of levels is greater than 40
        if lenUniqueVal > 40:
            n = 2 
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]           
                                
        # Bivariate box plot
        ax = p.makeCanvas(title = 'Faceted by target\n* {}'.format(feature), position = 122)
        p.prettyBoxPlotV(x = feature
                        ,y = self.target.name
                        ,data = biDf[biDf[feature].notnull()].sort_values([feature])
                        ,color = style.genCmap(len(uniqueVals), [style.styleHexMid[0], style.styleHexMid[1], style.styleHexMid[2]])
                        ,labelRotate = rotation
                        ,ax = ax
            )                   
        
        # hide every other label if total number of levels is greater than 40
        if lenUniqueVal > 40:
            n = 2 
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]           
        
        plt.show()

def dfSideBySide(self, dfs, names = []):
    """
    Documentation:
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

