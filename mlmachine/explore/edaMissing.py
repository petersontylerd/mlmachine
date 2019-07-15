import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from prettierplot.plotter import PrettierPlot
from prettierplot import style


def edaMissingSummary(self):
    """
    Documentation:
        Description:
            Creates vertical bar chart visualizating the percent of rows of a feature that is missing.
    """
    totalMissing = self.data.isnull().sum()
    percentMissing = self.data.isnull().sum() / len(self.data) * 100
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
            color=style.styleHexMid[2],
            yUnits="p",
            ax=ax,
        )
    else:
        print("No nulls")
