import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from prettierplot.plotter import PrettierPlot
from prettierplot import style


def eda_missing_summary(self, data=None, color=style.style_grey, chart_scale=25):
    """
    documentation:
        description:
            creates vertical bar chart visualizating the percent of rows of a feature that is missing.
        parameters:
            data : pandas DataFrame, default=None
                pandas DataFrame containing independent variables. if left as none,
                the feature dataset provided to machine during instantiation is used.
            color : string, color code, default=style.style_grey
                bar color.
            chart_scale : int or float, default=15
                controls chart size and proportions. higher value creates larger plots and increases
                visual elements proportionally.
    """
    # use data/target provided during instantiation if left unspecified
    if data is None:
        data = self.data

    # calcule missing data statistics
    total_missing = data.isnull().sum()
    percent_missing = data.isnull().sum() / len(data) * 100
    percent_missing = pd.DataFrame(
        {"total missing": total_missing, "percent missing": percent_missing}
    )
    percent_missing = percent_missing[
        ~(percent_missing["percent missing"].isna())
        & (percent_missing["percent missing"] > 0)
    ].sort_values(["percent missing"], ascending=False)

    if not percent_missing.empty:
        display(percent_missing)

        p = PrettierPlot(chart_scale=chart_scale)
        ax = p.make_canvas(
            title="percent missing by feature", y_shift=0.8, position=221
        )
        p.bar_v(
            x=percent_missing.index,
            counts=percent_missing["percent missing"],
            label_rotate=-90,
            color=color,
            y_units="p",
            ax=ax,
        )
    else:
        print("no nulls")
