import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
from scipy import special

from prettierplot.plotter import PrettierPlot
from prettierplot import style


def eda_missing_summary(self, training_data=True, color=style.style_grey, display_df=False, chart_scale=15):
    """
    Documentation:

        ---
        Description:
            Creates vertical bar chart visualizing the percent of values missing for each feature.
            Optionally displays the underlying Pandas DataFrame.

        ---
        Parameters:
            training_data : boolean, dafault=True
                Controls which dataset (training or validation) is used for visualization.
            color : str or color code, default=style.style_grey
                Bar color.
            display_df : boolean, default=False
                Controls whether to display summary data in Pandas DataFrame in addition to chart.
            chart_scale : int or float, default=15
                Controls size and proportions of chart and chart elements. Higher value creates
                larger plots and increases visual elements proportionally.
    """
    # dynamically choose training data objects or validation data objects
    data, _ = self.training_or_validation_dataset(training_data)
    
    # return missingness summary
    percent_missing = self.missing_summary(training_data)

    # if missingness summary is not empty, create the visualization
    if not percent_missing.empty:
        # optionally display DataFrame summary
        if display_df:
            display(percent_missing)

        # create prettierplot object
        p = PrettierPlot(chart_scale=chart_scale, plot_orientation="wide_standard")

        # add canvas to prettierplot object
        ax = p.make_canvas(
            title="Percent missing by feature",
            y_shift=0.8,
            title_scale=0.8,
        )

        # add vertical bar chart to canvas
        p.bar_v(
            x=percent_missing.index,
            counts=percent_missing["Percent missing"],
            label_rotate=45 if len(percent_missing.index) <=5 else 90,
            color=color,
            y_units="p",
            x_tick_wrap=False,
            ax=ax,
        )

        ax.set_ylim([0,100])

    # if missingness summary is empty, just print "No Nulls"
    else:
        print("No nulls")

def eda_skew_summary(self, training_data=True, color=style.style_grey, display_df=False, chart_scale=15):
    """
    Documentation:

        ---
        Description:
            Creates vertical bar chart visualizing the skew for each feature. Optionally
            displaying the underlying Pandas DataFrame.

        ---
        Parameters:
            training_data : boolean, dafault=True
                Controls which dataset (training or validation) is used for visualization.
            color : str, color code, default=style.style_grey
                Bar color.
            display_df : boolean, default=False
                Controls whether to display summary data in Pandas DataFrame along with chart.
            chart_scale : int or float, default=15
                Controls size and proportions of chart and chart elements. Higher value creates
                larger plots and increases visual elements proportionally.
    """
    # dynamically choose training data objects or validation data objects
    data, _ = self.training_or_validation_dataset(training_data)
    
    # return skewness summary
    skew_summary = self.skew_summary(data)

    # optionally display DataFrame summary
    if display_df:
        display(skew_summary)

    # create prettierplot object
    p = PrettierPlot(chart_scale=chart_scale, plot_orientation="wide_standard")

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title="Skew by feature",
        y_shift=0.8,
        title_scale=0.8,
    )

    # add vertical bar chart to canvas
    p.bar_v(
        x=skew_summary.index,
        counts=skew_summary["Skew"],
        label_rotate=45 if len(skew_summary.index) <=5 else 90,
        color=color,
        y_units="fff",
        x_tick_wrap=False,
        ax=ax,
    )

def eda_transform_target(self, data, name, chart_scale=15):
    """
    Documentation:

        ---
        Description:
            Creates a two_panel visualization. The left plot is the current distribution
            overlayed on a normal distribution. The right plot is a qqplot overlayed
            across a straight line.

        ---
        Parameters:
            data : Pandas Series
                Target variable data object.
            name : str
                Name of target variable.
            chart_scale : int or float, default=15
                Controls size and proportions of chart and chart elements. Higher value
                creates larger plots and increases visual elements proportionally.
    """
    # create prettierplot object
    p = PrettierPlot(chart_scale=chart_scale)

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title=f"dist/kde - {name} (initial)",
        x_label="",
        y_label="",
        y_shift=0.8,
        position=221,
    )

    # add distribution / kernel density plot to canvas
    p.dist_plot(
        data, color=style.style_grey, fit=stats.norm, x_rotate=True, ax=ax
    )

    # turn off x and y ticks
    plt.xticks([])
    plt.yticks([])

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title=f"probability plot - {name} (initial)",
        x_label="",
        y_label="",
        y_shift=0.8,
        position=222,
    )

    # add QQ / probability plot to canvas
    p.prob_plot(data, plot=ax)

    # turn off x and y ticks
    plt.xticks([])
    plt.yticks([])

def eda_transform_log1(self, data, name, chart_scale=15):
    """
    Documentation:

        ---
        Description:
            Creates a two_panel visualization. The left plot is the log + 1 transformed
            distribution overlayed on a normal distribution. The right plot is a log + 1
            adjusted qqplot overlayed across a straight line.

        ---
        Parameters:
            data : Pandas Series
                Target variable data object.
            name : str
                Name of target variable.
            chart_scale : int or float, default=15
                Controls size and proportions of chart and chart elements. Higher value creates
                larger plots and increases visual elements proportionally.
    """
    # create prettierplot object
    p = PrettierPlot(chart_scale=chart_scale)

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title=f"dist/kde - {name} (log+1)",
        x_label="",
        y_label="",
        y_shift=0.8,
        position=223,
    )

    # add distribution / kernel density plot to canvas
    p.dist_plot(
        np.log1p(data), color=style.style_grey, fit=stats.norm, x_rotate=True, ax=ax
    )

    # turn off x and y ticks
    plt.xticks([])
    plt.yticks([])

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title=f"probability plot - {name} (log+1)",
        x_label="",
        y_label="",
        y_shift=0.8,
        position=224,
    )

    # add QQ / probability plot to canvas
    p.prob_plot(np.log1p(data), plot=ax)

    # turn off x and y ticks
    plt.xticks([])
    plt.yticks([])

def eda_transform_box_cox(self, data, name, lmbda, chart_scale=15):
    """
    Documentation:

        ---
        Description:
            Creates a two_panel visualization. The left plot is the box-cox transformed
            distribution overlayed on a normal distribution. The right plot is a Box-Cox
            transformed qqplot overlayed across a straight line.

        ---
        Parameters:
            data : Pandas Series
                Target variable data object.
            name : str
                Name of target variable.
            lmbda : float
                Box-Cox transformation parameter.
            chart_scale : int or float, default=15
                Controls size and proportions of chart and chart elements. Higher value
                creates larger plots and increases visual elements proportionally.
    """
    # create prettierplot object
    p = PrettierPlot(chart_scale=chart_scale)

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title=f"dist/kde - {name} (box-cox, {lmbda})",
        x_label="",
        y_label="",
        y_shift=0.8,
        position=223,
    )

    # add distribution / kernel density plot to canvas
    p.dist_plot(
        special.boxcox1p(data, lmbda),
        color=style.style_grey,
        fit=stats.norm,
        x_rotate=True,
        ax=ax,
    )

    # turn off x and y ticks
    plt.xticks([])
    plt.yticks([])

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title=f"Probability plot - {name} (box-cox, {lmbda})",
        x_label="",
        y_label="",
        y_shift=0.8,
        position=224,
    )

    # add QQ / probability plot to canvas
    p.prob_plot(special.boxcox1p(data, lmbda), plot=ax)

    # turn off x and y ticks
    plt.xticks([])
    plt.yticks([])