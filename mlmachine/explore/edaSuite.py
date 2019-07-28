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


def edaCatTargetCatFeat(self, skipCols=[]):
    """
    Documentation:
        Description:
            Produces exploratory data visualizations and statistical summaries for all categorical
            features in the context of a categorical target.
        Parameters
            skipCols : list, default = []
                Column to skip over in visualization creation loop.
    """
    # iterate through each feature within a feature type
    for feature in self.featureByDtype_["categorical"]:
        if feature not in skipCols:
            # univariate summary
            uniSummDf = pd.DataFrame(columns=[feature, "Count", "Proportion"])
            uniqueVals, uniqueCounts = np.unique(
                self.data[self.data[feature].notnull()][feature], return_counts=True
            )
            for i, j in zip(uniqueVals, uniqueCounts):
                uniSummDf = uniSummDf.append(
                    {
                        feature: i,
                        "Count": j,
                        "Proportion": j / np.sum(uniqueCounts) * 100,
                    },
                    ignore_index=True,
                )
            uniSummDf = uniSummDf.sort_values(by=["Proportion"], ascending=False)

            # bivariate summary
            biDf = pd.concat([self.data[feature], self.target], axis=1)

            biSummDf = (
                biDf.groupby([feature] + [self.target.name])
                .size()
                .reset_index()
                .pivot(columns=self.target.name, index=feature, values=0)
            )

            multiIndex = biSummDf.columns
            singleIndex = pd.Index([i for i in multiIndex.tolist()])
            biSummDf.columns = singleIndex
            biSummDf.reset_index(inplace=True)

            # add PercentPositive column
            if len(np.unique(self.target)):
                biSummDf["PercentPositive"] = (
                    biSummDf[1] / (biSummDf[1] + biSummDf[0])
                ) * 100

            # execute z-test
            if len(np.unique(biDf[biDf[feature].notnull()][feature])) == 2:

                # total observations
                totalObs1 = biDf[(biDf[feature] == np.unique(biDf[feature])[0])][
                    feature
                ].shape[0]
                totalObs2 = biDf[(biDf[feature] == np.unique(biDf[feature])[1])][
                    feature
                ].shape[0]

                # total positive observations
                posObs1 = biDf[
                    (biDf[feature] == np.unique(biDf[feature])[0])
                    & (biDf[self.target.name] == 1)
                ][feature].shape[0]
                posObs2 = biDf[
                    (biDf[feature] == np.unique(biDf[feature])[1])
                    & (biDf[self.target.name] == 1)
                ][feature].shape[0]

                z, pVal = proportions_ztest(
                    count=(posObs1, posObs2), nobs=(totalObs1, totalObs2)
                )

                statTestDf = pd.DataFrame(
                    data=[{"z-test statistic": z, "p-value": pVal}],
                    columns=["z-test statistic", "p-value"],
                    index=[feature],
                ).round(4)

                # display summary tables
                self.dfSideBySide(
                    dfs=(uniSummDf, biSummDf, statTestDf),
                    names=[
                        "Univariate summary",
                        "Biivariate summary",
                        "Statistical test",
                    ],
                )
                if "PercentPositive" in biSummDf:
                    biSummDf = biSummDf.drop(["PercentPositive"], axis=1)

            else:
                # display summary tables
                self.dfSideBySide(
                    dfs=(uniSummDf, biSummDf),
                    names=["Univariate summary", "Biivariate summary"],
                )
                if "PercentPositive" in biSummDf:
                    biSummDf = biSummDf.drop(["PercentPositive"], axis=1)

            # set label rotation angle
            lenUniqueVal = len(uniqueVals)
            avgLenUniqueVal = sum(map(len, str(uniqueVals))) / len(uniqueVals)
            if lenUniqueVal <= 4 and avgLenUniqueVal <= 12:
                rotation = 0
            elif lenUniqueVal >= 5 and lenUniqueVal <= 8 and avgLenUniqueVal <= 8:
                rotation = 0
            elif lenUniqueVal >= 9 and lenUniqueVal <= 14 and avgLenUniqueVal <= 4:
                rotation = 0
            else:
                rotation = 90

            # instantiate charting object
            p = PrettierPlot(chartProp=15, plotOrientation="wide")

            # univariate plot
            ax = p.makeCanvas(title="Univariate\n* {}".format(feature), position=121)

            p.prettyBarV(
                x=list(map(str, uniqueVals.tolist())),
                counts=uniqueCounts,
                labelRotate=rotation,
                color=style.styleHexMid[2],
                yUnits="f",
                ax=ax,
            )
            
            # bivariate plot
            ax = p.makeCanvas(
                title="Faceted by target\n* {}".format(feature), position=122
            )
            p.prettyFacetCat(df=biSummDf, feature=feature, labelRotate=rotation, ax=ax)
            plt.show()


def edaCatTargetNumFeat(self, skipCols=[]):
    """
    Documentation:
        Description:
            Produces exploratory data visualizations and statistical summaries for all continuous
            features in the context of a categorical target.
        Parameters
            skipCols : list, default = []
                Columns to skip over in visualization creation loop.
    """
    # Iterate through each feature within a feature type
    for feature in self.featureByDtype_["continuous"]:
        if feature not in skipCols:
            # bivariate roll-up table
            biDf = pd.concat([self.data[feature], self.target], axis=1)
            
            # bivariate summary statistics
            biSummStatsDf = pd.DataFrame(
                columns=[feature, "Count", "Proportion", "Mean", "StdDv"]
            )

            for labl in np.unique(self.target):
                featureSlice = biDf[biDf[self.target.name] == labl][feature]

                biSummStatsDf = biSummStatsDf.append(
                    {
                        feature: labl,
                        "Count": len(featureSlice),
                        "Proportion": len(featureSlice) / len(biDf[feature]) * 100,
                        "Mean": np.mean(featureSlice),
                        "StdDv": np.std(featureSlice),
                    },
                    ignore_index=True,
                )

            # display summary tables
            describeDf = pd.DataFrame(biDf[feature].describe()).reset_index()
            describeDf = describeDf.append(
                {
                    "index": "skew",
                    feature: np.round(
                        stats.skew(biDf[feature].values, nan_policy="omit"), 5
                    ),
                },
                ignore_index=True,
            )
            describeDf = describeDf.append(
                {
                    "index": "kurtosis",
                    feature: stats.kurtosis(biDf[feature].values, nan_policy="omit"),
                },
                ignore_index=True,
            )
            describeDf = describeDf.rename(columns={"index": ""})

            # execute z-test or t-test
            if len(np.unique(self.target)) == 2:
                s1 = biDf[
                    (biDf[self.target.name] == biDf[self.target.name].unique()[0])
                    & (biDf[feature].notnull())
                ][feature]
                s2 = biDf[
                    (biDf[self.target.name] == biDf[self.target.name].unique()[1])
                    & (biDf[feature].notnull())
                ][feature]
                if len(s1) > 30 and len(s2) > 30:
                    z, pVal = ztest(s1, s2)

                    statTestDf = pd.DataFrame(
                        data=[{"z-test statistic": z, "p-value": pVal}],
                        columns=["z-test statistic", "p-value"],
                        index=[feature],
                    ).round(4)
                else:
                    t, pVal = stats.ttest_ind(s1, s2)

                    statTestDf = pd.DataFrame(
                        data=[{"t-test statistic": t, "p-value": pVal}],
                        columns=["t-test statistic", "p-value"],
                        index=[feature],
                    ).round(4)
                self.dfSideBySide(
                    dfs=(describeDf, biSummStatsDf, statTestDf),
                    names=["Univariate stats", "bivariate stats", "Statistical test"],
                )
            else:
                self.dfSideBySide(
                    dfs=(describeDf, biSummStatsDf),
                    names=["Descriptive stats", "bivariate stats"],
                )

            # instantiate charting object
            p = PrettierPlot(chartProp=15, plotOrientation="wide")

            # univariate plot
            ax = p.makeCanvas(
                title="Dist/KDE - Univariate\n* {}".format(feature), position=151
            )
            p.prettyDistPlot(
                biDf[(biDf[feature].notnull())][feature].values,
                color=style.styleHexMid[2],
                yUnits="f",
                ax=ax,
            )

            # probability plot
            ax = p.makeCanvas(title="Probability plot\n* {}".format(feature), position=152)
            p.prettyProbPlot(x=biDf[(biDf[feature].notnull())][feature].values, plot=ax)

            # bivariate kernel density plot
            ax = p.makeCanvas(
                title="KDE - Faceted by target\n* {}".format(feature), position=153
            )

            for ix, labl in enumerate(
                np.unique(biDf[(biDf[feature].notnull())][self.target.name].values)
            ):
                p.prettyKdePlot(
                    biDf[(biDf[feature].notnull()) & (biDf[self.target.name] == labl)][
                        feature
                    ].values,
                    color=style.styleHexMid[ix],
                    yUnits="ffff",
                    ax=ax,
                )

            # bivariate histogram
            ax = p.makeCanvas(
                title="Hist - Faceted by target\n* {}".format(feature), position=154
            )
            for ix, labl in enumerate(
                np.unique(biDf[(biDf[feature].notnull())][self.target.name].values)
            ):
                p.prettyHist(
                    biDf[(biDf[feature].notnull()) & (biDf[self.target.name] == labl)][
                        feature
                    ].values,
                    color=style.styleHexMid[ix],
                    label=labl,
                    alpha=0.4,
                )

            # boxplot histogram
            ax = p.makeCanvas(
                title="Boxplot - Faceted by target\n* {}".format(feature), position=155
            )
            p.prettyBoxPlotH(x=feature, y=self.target.name, data=biDf, ax=ax)
            plt.show()


def edaNumTargetNumFeat(self, skipCols=[]):
    """
    Documentation:
        Description:
            Produces exploratory data visualizations and statistical summaries for all continuous
            features in the context of a continuous target.
        Parameters
            skipCols : list, default = []
                Columns to skip over in visualization creation loop.
    """
    # iterate through each feature within a feature type
    for feature in self.featureByDtype_["continuous"]:
        if feature not in skipCols:
            ### Summary tables
            # Define bivariate dataframe
            biDf = pd.concat([self.data[feature], self.target], axis=1)

            biDf[self.target.name] = biDf[self.target.name].astype(float)

            # define summary tables
            describeDf = pd.DataFrame(biDf[feature].describe()).reset_index()

            # instantiate charting object
            p = PrettierPlot(chartProp=15, plotOrientation="wide")

            # if continuous variable has fewer than a set number of unique variables, represent variable
            # as a categorical variable vs. a continuous target variable
            if len(np.unique(biDf[feature].values)) <= 20:

                describeDf = describeDf.rename(columns={"index": ""})

                # bivariate summary statistics
                biSummStatsDf = pd.DataFrame(
                    columns=[feature, "Count", "Proportion", "Mean", "StdDv"]
                )
                uniqueVals, uniqueCounts = np.unique(
                    self.data[self.data[feature].notnull()][feature], return_counts=True
                )
                for featureVal in np.unique(biDf[feature].values):
                    featureSlice = biDf[
                        (biDf[feature] == featureVal) & (biDf[feature].notnull())
                    ][feature]

                    biSummStatsDf = biSummStatsDf.append(
                        {
                            feature: featureVal,
                            "Count": len(featureSlice),
                            "Proportion": len(featureSlice) / len(biDf[feature]) * 100,
                            "Mean": np.mean(featureSlice),
                            "StdDv": np.std(featureSlice),
                        },
                        ignore_index=True,
                    )

                # display summary dataframes
                self.dfSideBySide(
                    dfs=(describeDf, biSummStatsDf),
                    names=["Univariate stats", "bivariate stats"],
                )

                # set rotation angle
                lenUniqueVal = len(uniqueVals)
                avgLenUniqueVal = sum(map(len, str(uniqueVals))) / len(uniqueVals)
                if lenUniqueVal <= 4 and avgLenUniqueVal <= 12:
                    rotation = 0
                elif lenUniqueVal >= 5 and lenUniqueVal <= 8 and avgLenUniqueVal <= 8:
                    rotation = 0
                elif lenUniqueVal >= 9 and lenUniqueVal <= 14 and avgLenUniqueVal <= 4:
                    rotation = 0
                else:
                    rotation = 90

                # univariate plot
                ax = p.makeCanvas(title="Univariate\n* {}".format(feature), position=131)
                p.prettyBarV(
                    x=list(map(str, uniqueVals.tolist())),
                    counts=uniqueCounts,
                    labelRotate=rotation,
                    color=style.styleHexMid[2],
                    yUnits="f",
                    ax=ax,
                )

                # regression plot
                ax = p.makeCanvas(
                    title="Regression plot\n* {}".format(feature), position=132
                )
                p.prettyRegPlot(
                    x=feature,
                    y=self.target.name,
                    data=biDf[biDf[feature].notnull()],
                    x_jitter=0.2,
                    ax=ax,
                )

                # hide every other label if total number of levels is less than 5
                if lenUniqueVal <= 4:
                    xmin, xmax = ax.get_xlim()
                    ax.set_xticks(np.round(np.linspace(xmin, xmax, lenUniqueVal), 2))

                # bivariate box plot
                ax = p.makeCanvas(
                    title="Box plot - Faceted by\n* {}".format(feature), position=133
                )
                p.prettyBoxPlotV(
                    x=feature,
                    y=self.target.name,
                    data=biDf[biDf[feature].notnull()],
                    color=style.genCmap(
                        len(uniqueVals),
                        [style.styleHexMid[0], style.styleHexMid[1], style.styleHexMid[2]],
                    ),
                    labelRotate=rotation,
                    ax=ax,
                )

            # if continuous variable has greater than a set number of unique variables, represent variable
            # as a continuous variable vs. a continuous target variable
            else:

                # add skew and curtosis to describeDf
                describeDf = describeDf.append(
                    {
                        "index": "skew",
                        feature: stats.skew(biDf[feature].values, nan_policy="omit"),
                    },
                    ignore_index=True,
                )
                describeDf = describeDf.append(
                    {
                        "index": "kurtosis",
                        feature: stats.kurtosis(biDf[feature].values, nan_policy="omit"),
                    },
                    ignore_index=True,
                )
                describeDf = describeDf.rename(columns={"index": ""})

                # display summary dataframes
                display(describeDf)

                # univariate plot
                ax = p.makeCanvas(
                    title="Dist/KDE - Univariate\n* {}".format(feature), position=131
                )
                p.prettyDistPlot(
                    biDf[(biDf[feature].notnull())][feature].values,
                    color=style.styleHexMid[2],
                    yUnits="ffff",
                    fit=stats.norm,
                    xRotate=45,
                    ax=ax,
                )

                # probability plot
                ax = p.makeCanvas(
                    title="Probability plot\n* {}".format(feature), position=132
                )
                p.prettyProbPlot(x=biDf[(biDf[feature].notnull())][feature].values, plot=ax)

                # regression plot
                ax = p.makeCanvas(
                    title="Regression plot\n* {}".format(feature), position=133
                )
                p.prettyRegPlot(
                    x=feature,
                    y=self.target.name,
                    data=biDf[biDf[feature].notnull()],
                    x_jitter=0.1,
                    xRotate=45,
                    ax=ax,
                )
            plt.show()


def edaNumTargetCatFeat(self, skipCols=[]):
    """
    Documentation:
        Description:
            Produces exploratory data visualizations and statistical summaries for all continuous
            features in the context of a categorical target.
        Parameters
            skipCols : list, default = []
                Columns to skip over in visualization creation loop.
    """
    # iterate through each feature within a feature type
    for feature in self.featureByDtype_["categorical"]:
        if feature not in skipCols:
            ### summary tables
            # univariate summary
            uniSummDf = pd.DataFrame(columns=[feature, "Count", "Proportion"])
            uniqueVals, uniqueCounts = np.unique(
                self.data[self.data[feature].notnull()][feature], return_counts=True
            )
            for i, j in zip(uniqueVals, uniqueCounts):
                uniSummDf = uniSummDf.append(
                    {feature: i, "Count": j, "Proportion": j / np.sum(uniqueCounts) * 100},
                    ignore_index=True,
                )
            uniSummDf = uniSummDf.sort_values(by=["Proportion"], ascending=False)

            # bivariate summary
            biDf = pd.concat([self.data[feature], self.target], axis=1)

            biDf[self.target.name] = biDf[self.target.name].astype(float)
            statsDict = {
                "N": len,
                "Median": np.nanmedian,
                "Mean": np.nanmean,
                "StdDev": np.nanstd,
            }
            biSummPivDf = pd.pivot_table(
                biDf, index=feature, aggfunc={self.target.name: statsDict}
            )
            multiIndex = biSummPivDf.columns
            singleIndex = pd.Index([i[1] for i in multiIndex.tolist()])
            biSummPivDf.columns = singleIndex
            biSummPivDf.reset_index(inplace=True)

            # display summary tables
            self.dfSideBySide(
                dfs=(uniSummDf, biSummPivDf),
                names=["Univariate summary", "bivariate summary"],
            )

            ### plots
            # instantiate charting object
            p = PrettierPlot(chartProp=15, plotOrientation="wide")

            # univariate plot
            ax = p.makeCanvas(title="Univariate\n* {}".format(feature), position=121)

            # select error catching block for resorting labels
            try:
                sorted(uniqueVals, key=int)
            except ValueError:
                pass
            else:
                # sort uniqueVals/uniqueCounts for bar chart
                newIx = [
                    sorted(list(uniqueVals), key=int).index(i) for i in list(uniqueVals)
                ]
                uniqueVals = np.array(sorted(list(uniqueVals), key=int))
                uniqueCounts = np.array([y for x, y in sorted(zip(newIx, uniqueCounts))])
                
                # sort temporary data frame for box plot
                biDf[feature] = biDf[feature].astype(int)

            # set rotation angle
            lenUniqueVal = len(uniqueVals)
            avgLenUniqueVal = sum(map(len, str(uniqueVals))) / len(uniqueVals)
            if lenUniqueVal <= 4 and avgLenUniqueVal <= 12:
                rotation = 0
            elif lenUniqueVal >= 5 and lenUniqueVal <= 8 and avgLenUniqueVal <= 10:
                rotation = 0
            elif lenUniqueVal >= 9 and lenUniqueVal <= 14 and avgLenUniqueVal <= 6:
                rotation = 0
            else:
                rotation = 90

            p.prettyBarV(
                x=list(map(str, uniqueVals.tolist())),
                counts=uniqueCounts,
                labelRotate=rotation,
                color=style.styleHexMid[2],
                yUnits="f",
                ax=ax,
            )

            # hide every other label if total number of levels is greater than 40
            if lenUniqueVal > 40:
                n = 2
                [
                    l.set_visible(False)
                    for (i, l) in enumerate(ax.xaxis.get_ticklabels())
                    if i % n != 0
                ]

            # bivariate box plot
            ax = p.makeCanvas(title="Faceted by target\n* {}".format(feature), position=122)
            p.prettyBoxPlotV(
                x=feature,
                y=self.target.name,
                data=biDf[biDf[feature].notnull()].sort_values([feature]),
                color=style.genCmap(
                    len(uniqueVals),
                    [style.styleHexMid[0], style.styleHexMid[1], style.styleHexMid[2]],
                ),
                labelRotate=rotation,
                ax=ax,
            )

            # hide every other label if total number of levels is greater than 40
            if lenUniqueVal > 40:
                n = 2
                [
                    l.set_visible(False)
                    for (i, l) in enumerate(ax.xaxis.get_ticklabels())
                    if i % n != 0
                ]

            plt.show()


def dfSideBySide(self, dfs, names=[]):
    """
    Documentation:
        Description:
            Helper function for displaying Pandas DataFrames side by side in a 
            notebook.
        Parameters:
            dfs : list
                List of dfs to be displayed
            names : list, default = []
                List of names to be displayed above dataframes.
    """
    html_str = ""
    if names:
        html_str += (
            "<tr>"
            + "".join(f'<td style="text-align:center">{name}</td>' for name in names)
            + "</tr>"
        )
    html_str += (
        "<tr>"
        + "".join(
            f'<td style="vertical-align:top"> {df.to_html(index=False)}</td>'
            for df in dfs
        )
        + "</tr>"
    )
    html_str = f"<table>{html_str}</table>"
    html_str = html_str.replace("table", 'table style="display:inline"')
    display_html(html_str, raw=True)

