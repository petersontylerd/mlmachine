import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
from scipy import special

from prettierplot.plotter import PrettierPlot
from prettierplot import style


def eda_transform_initial(self, data, name):
    """
    documentation:
        description:
            creates a two_panel visualization. the left plot is the current distribution overlayed on a
            normal distribution. the right plot is a qqplot overlayed across a straight line.
        parameters:
            data : Pandas Series
                target variables data object.
            name : string
                name of target variable.
    """
    p = PrettierPlot(chart_prop=15)

    # distribution / kernel density plot
    ax = p.make_canvas(
        title="dist/kde - {} (initial)".format(name),
        x_label="",
        y_label="",
        y_shift=0.8,
        position=221,
    )
    p.pretty_dist_plot(
        data, color=style.style_grey, fit=stats.norm, x_rotate=True, ax=ax
    )
    plt.xticks([])
    plt.yticks([])

    # qq plot
    ax = p.make_canvas(
        title="probability plot - {} (initial)".format(name),
        x_label="",
        y_label="",
        y_shift=0.8,
        position=222,
    )
    p.pretty_prob_plot(data, plot=ax)
    plt.xticks([])
    plt.yticks([])


def eda_transform_log1(self, data, name):
    """
    documentation:
        description:
            creates a two_panel visualization. the left plot is the log + 1 adjusted distribution overlayed
            on a normal distribution. the right plot is a log + 1 adjusted qqplot overlayed across a straight
            line.
        parameters:
            data : Pandas Series
                target variables data object.
            name : string
                name of target variable.
    """
    p = PrettierPlot(chart_prop=15)

    # distribution / kernel density plot
    ax = p.make_canvas(
        title="dist/kde - {} (log+1)".format(name),
        x_label="",
        y_label="",
        y_shift=0.8,
        position=223,
    )
    p.pretty_dist_plot(
        np.log1p(data), color=style.style_grey, fit=stats.norm, x_rotate=True, ax=ax
    )
    plt.xticks([])
    plt.yticks([])

    # qq plot
    ax = p.make_canvas(
        title="probability plot - {} (log+1)".format(name),
        x_label="",
        y_label="",
        y_shift=0.8,
        position=224,
    )
    p.pretty_prob_plot(np.log1p(data), plot=ax)
    plt.xticks([])
    plt.yticks([])


def eda_transform_box_cox(self, data, name, lmbda):
    """
    documentation:
        description:
            creates a two_panel visualization. the left plot is the box-cox transformed distribution overlayed
            on a normal distribution. the right plot is a box-cox transformed qqplot overlayed across a straight
            line.
        parameters:
            data : Pandas Series
                target variables data object.
            name : string
                name of target variable.
            lmbda : float
                box-cox transformation parameter.
    """
    p = PrettierPlot(chart_prop=15)

    # distribution / kernel density plot
    ax = p.make_canvas(
        title="dist/kde - {} (box-cox, {})".format(name, lmbda),
        x_label="",
        y_label="",
        y_shift=0.8,
        position=223,
    )
    p.pretty_dist_plot(
        special.boxcox1p(data, lmbda),
        color=style.style_grey,
        fit=stats.norm,
        x_rotate=True,
        ax=ax,
    )
    plt.xticks([])
    plt.yticks([])

    # qq plot
    ax = p.make_canvas(
        title="probability plot - {} (box-cox, {})".format(name, lmbda),
        x_label="",
        y_label="",
        y_shift=0.8,
        position=224,
    )
    p.pretty_prob_plot(special.boxcox1p(data, lmbda), plot=ax)
    plt.xticks([])
    plt.yticks([])

