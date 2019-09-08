import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
from scipy import special

from prettierplot.plotter import PrettierPlot
from prettierplot import style


def edaTransformInitial(self, data, name):
    """
    Documentation:
        Description:
            Creates a two-panel visualization. The left plot is the current distribution overlayed on a
            normal distribution. The right plot is a qqplot overlayed across a straight line.
        Parameters:
            data : Pandas Series
                Target variables data object.
            name : string
                Name of target variable.
    """
    p = PrettierPlot(chartProp=15)

    # distribution / kernel density plot
    ax = p.makeCanvas(
        title="Dist/KDE - {} (Initial)".format(name),
        xLabel="",
        yLabel="",
        yShift=0.8,
        position=221,
    )
    p.prettyDistPlot(
        data, color=style.styleGrey, fit=stats.norm, xRotate=True, ax=ax
    )
    plt.xticks([])
    plt.yticks([])

    # QQ plot
    ax = p.makeCanvas(
        title="Probability plot - {} (Initial)".format(name),
        xLabel="",
        yLabel="",
        yShift=0.8,
        position=222,
    )
    p.prettyProbPlot(data, plot=ax)
    plt.xticks([])
    plt.yticks([])


def edaTransformLog1(self, data, name):
    """
    Documentation:
        Description:
            Creates a two-panel visualization. The left plot is the log + 1 adjusted distribution overlayed
            on a normal distribution. The right plot is a log + 1 adjusted qqplot overlayed across a straight
            line.
        Parameters:
            data : Pandas Series
                Target variables data object.
            name : string
                Name of target variable.
    """
    p = PrettierPlot(chartProp=15)

    # distribution / kernel density plot
    ax = p.makeCanvas(
        title="Dist/KDE - {} (log+1)".format(name),
        xLabel="",
        yLabel="",
        yShift=0.8,
        position=223,
    )
    p.prettyDistPlot(
        np.log1p(data), color=style.styleGrey, fit=stats.norm, xRotate=True, ax=ax
    )
    plt.xticks([])
    plt.yticks([])

    # QQ plot
    ax = p.makeCanvas(
        title="Probability plot - {} (log+1)".format(name),
        xLabel="",
        yLabel="",
        yShift=0.8,
        position=224,
    )
    p.prettyProbPlot(np.log1p(data), plot=ax)
    plt.xticks([])
    plt.yticks([])


def edaTransformBoxCox(self, data, name, lmbda):
    """
    Documentation:
        Description:
            Creates a two-panel visualization. The left plot is the Box-Cox transformed distribution overlayed
            on a normal distribution. The right plot is a Box-Cox transformed qqplot overlayed across a straight
            line.
        Parameters:
            data : Pandas Series
                Target variables data object.
            name : string
                Name of target variable.
            lmbda : float
                Box-Cox transformation parameter.
    """
    p = PrettierPlot(chartProp=15)

    # distribution / kernel density plot
    ax = p.makeCanvas(
        title="Dist/KDE - {} (BoxCox, {})".format(name, lmbda),
        xLabel="",
        yLabel="",
        yShift=0.8,
        position=223,
    )
    p.prettyDistPlot(
        special.boxcox1p(data, lmbda),
        color=style.styleGrey,
        fit=stats.norm,
        xRotate=True,
        ax=ax,
    )
    plt.xticks([])
    plt.yticks([])

    # QQ plot
    ax = p.makeCanvas(
        title="Probability plot - {} (BoxCox, {})".format(name, lmbda),
        xLabel="",
        yLabel="",
        yShift=0.8,
        position=224,
    )
    p.prettyProbPlot(special.boxcox1p(data, lmbda), plot=ax)
    plt.xticks([])
    plt.yticks([])

