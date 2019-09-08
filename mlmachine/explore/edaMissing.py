import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from prettierplot.plotter import PrettierPlot
from prettierplot import style


def edaMissingSummary(self, data=None, color=style.styleGrey):
    """
    Documentation:
        Description:
            Creates vertical bar chart visualizating the percent of rows of a feature that is missing.
        Parameters:
            data : Pandas DataFrame, default = None
                Pandas DataFrame containing independent variables. If left as None,
                the feature dataset provided to Machine during instantiation is used.
            color : string, color code, default = style.styleGrey
                Bar color.
    """
    # use data/target provided during instantiation if left unspecified
    if data is None:
        data = self.data

    # calcule missing data statistics
    totalMissing = data.isnull().sum()
    percentMissing = data.isnull().sum() / len(data) * 100
    percentMissing = pd.DataFrame(
        {"Total missing": totalMissing, "Percent missing": percentMissing}
    )
    percentMissing = percentMissing[
        ~(percentMissing["Percent missing"].isna())
        & (percentMissing["Percent missing"] > 0)
    ].sort_values(["Percent missing"], ascending=False)

    if not percentMissing.empty:
        display(percentMissing)

        p = PrettierPlot(chartProp=15)
        ax = p.makeCanvas(title="Percent missing by feature", yShift=0.8, position=221)
        p.prettyBarV(
            x=percentMissing.index,
            counts=percentMissing["Percent missing"],
            labelRotate=90,
            color=color,
            yUnits="p",
            ax=ax,
        )
    else:
        print("No nulls")