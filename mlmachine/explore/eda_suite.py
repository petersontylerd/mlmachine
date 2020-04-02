import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
import numpy as np

import seaborn as sns
import squarify
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from statsmodels.stats.weightstats import ztest
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

from IPython.display import display_html

import os
import sys

from prettierplot.plotter import PrettierPlot
from prettierplot import style



def eda_cat_target_cat_feat(self, feature, level_count_cap=50, color_map="viridis", legend_labels=None,
                            chart_scale=15):
    """
    Documentation:

        ---
        Description:
            Creates exploratory data visualizations and statistical summaries for a category feature
            in the context of a categorical target.

        ---
        Parameters:
            feature : string
                Feature to visualize.
            level_count_cap : int, default=50
                Maximum number of unique levels in feature. If the number of levels exceeds the
                cap, then no visualization panel is produced.
            color_map : string specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            legend_labels : list, default=None
                Class labels displayed in plot legend.
            chart_scale : int or float, default=15
                Controls size and proportions of chart and chart elements. Higher value creates
                larger plots and increases visual elements proportionally.
    """

    # if number of unique levels in feature is less than specified level_count_cap
    if (len(np.unique(self.data[self.data[feature].notnull()][feature].values)) < level_count_cap):

        ### data summaries
        ## feature summary
        # create empty DataFrame
        uni_summ_df = pd.DataFrame(columns=[feature, "Count", "Proportion"])

        # capture unique values and count of those unique values
        unique_vals, unique_counts = np.unique(
            self.data[self.data[feature].notnull()][feature], return_counts=True
        )

        # append each unique value, count and proportion to DataFrame
        for i, j in zip(unique_vals, unique_counts):
            uni_summ_df = uni_summ_df.append(
                {
                    feature: i,
                    "Count": j,
                    "Proportion": j / np.sum(unique_counts) * 100,
                },
                ignore_index=True,
            )

        # sort DataFrame by "Proportion", descending
        uni_summ_df = uni_summ_df.sort_values(by=["Proportion"], ascending=False)

        # set values to int dtype where applicable to optimize
        uni_summ_df["Count"] = uni_summ_df["Count"].astype("int64")
        if is_numeric_dtype(uni_summ_df[feature]):
            uni_summ_df[feature] = uni_summ_df[feature].astype("int64")

        ## feature vs. target summary
        # combine feature column and target
        bi_df = pd.concat([self.data[feature], self.target], axis=1)

        # remove any rows with nulls
        bi_df = bi_df[bi_df[feature].notnull()]

        # groupby category feature and count the occurrences of target classes
        # for each level in category
        bi_summ_df = (
            bi_df.groupby([feature] + [self.target.name])
            .size()
            .reset_index()
            .pivot(columns=self.target.name, index=feature, values=0)
        )

        # overwrite DataFrame index with actual class labels if provided
        bi_summ_df.columns = pd.Index(legend_labels) if legend_labels is not None else pd.Index([i for i in bi_summ_df.columns.tolist()])
        bi_summ_df.reset_index(inplace=True)

        # fill nan's with zero
        fill_columns = bi_summ_df.iloc[:,2:].columns
        bi_summ_df[fill_columns] = bi_summ_df[fill_columns].fillna(0)

        # set values to int dtype where applicable to optimize displayed DataFrame
        for column in bi_summ_df.columns:
            try:
                bi_summ_df[column] = bi_summ_df[column].astype(np.int)
            except ValueError:
                bi_summ_df[column] = bi_summ_df[column]

        ## proportion by category summary
        # combine feature column and target
        prop_df = pd.concat([self.data[feature], self.target], axis=1)

        # remove any rows with nulls
        prop_df = prop_df[prop_df[feature].notnull()]

        # calculate percent of 100 by class label
        prop_df = prop_df.groupby([feature, self.target.name]).agg({self.target.name : {"count"}})
        prop_df = prop_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
        prop_df = prop_df.reset_index()

        multiIndex = prop_df.columns
        singleIndex = [i[0] for i in multiIndex.tolist()]
        singleIndex[-1] = "Count"
        prop_df.columns = singleIndex
        prop_df = prop_df.reset_index(drop=True)

        prop_df = pd.pivot_table(prop_df, values=["Count"], columns=[feature], index=[self.target.name], aggfunc={"Count": np.mean})
        prop_df = prop_df.reset_index(drop=True)

        multiIndex = prop_df.columns
        singleIndex = []

        for column in multiIndex.tolist():
            try:
                singleIndex.append(int(column[1]))
            except ValueError:
                singleIndex.append(column[1])

        prop_df.columns = singleIndex
        prop_df = prop_df.reset_index(drop=True)

        # insert column to DataFrame with actual class labels if provided, otherwise use raw class labels in target
        prop_df.insert(loc=0, column="Class", value=legend_labels if legend_labels is not None else np.unique(self.target))

        # fill nan's with zero
        fill_columns = prop_df.iloc[:,:].columns
        prop_df[fill_columns] = prop_df[fill_columns].fillna(0)

        # if there are only two class labels, perform z-test/t-test
        if len(np.unique(bi_df[bi_df[feature].notnull()][feature])) == 2:

            # total observations
            total_obs1 = bi_df[(bi_df[feature] == np.unique(bi_df[feature])[0])][
                feature
            ].shape[0]
            total_obs2 = bi_df[(bi_df[feature] == np.unique(bi_df[feature])[1])][
                feature
            ].shape[0]

            # total positive observations
            pos_obs1 = bi_df[
                (bi_df[feature] == np.unique(bi_df[feature])[0])
                & (bi_df[self.target.name] == 1)
            ][feature].shape[0]
            pos_obs2 = bi_df[
                (bi_df[feature] == np.unique(bi_df[feature])[1])
                & (bi_df[self.target.name] == 1)
            ][feature].shape[0]

            # perform z-test, return z-statistic and p-value
            z, p_val = proportions_ztest(
                count=(pos_obs1, pos_obs2), nobs=(total_obs1, total_obs2)
            )

            # add z-statistic and p-value to DataFrame
            stat_test_df = pd.DataFrame(
                data=[{"z-test statistic": z, "p-value": p_val}],
                columns=["z-test statistic", "p-value"],
                index=[feature],
            ).round(4)

            # display summary tables
            self.df_side_by_side(
                dfs=(uni_summ_df, bi_summ_df, prop_df, stat_test_df),
                names=["Feature summary", "Feature vs. target summary", "Target proportion", "Statistical test",],
            )
            if "percent_positive" in bi_summ_df:
                bi_summ_df = bi_summ_df.drop(["percent_positive"], axis=1)

        else:
            # display summary tables
            self.df_side_by_side(
                dfs=(uni_summ_df, bi_summ_df, prop_df),
                names=["Feature summary", "Feature vs. target summary", "Target proportion"],
            )
            if "percent_positive" in bi_summ_df:
                bi_summ_df = bi_summ_df.drop(["percent_positive"], axis=1)

        ### visualizations
        # set label rotation angle
        len_unique_val = len(unique_vals)
        avg_len_unique_val = sum(map(len, str(unique_vals))) / len(unique_vals)
        if len_unique_val <= 4 and avg_len_unique_val <= 12:
            rotation = 0
        elif len_unique_val >= 5 and len_unique_val <= 8 and avg_len_unique_val <= 8:
            rotation = 0
        elif len_unique_val >= 9 and len_unique_val <= 14 and avg_len_unique_val <= 4:
            rotation = 0
        else:
            rotation = 90

        # create prettierplot object
        p = PrettierPlot(chart_scale=chart_scale, plot_orientation="wide_narrow")

        # add canvas to prettierplot object
        ax = p.make_canvas(title="Category counts\n* {}".format(feature), position=131, title_scale=0.82)

        # add treemap to canvas
        p.tree_map(
            counts=uni_summ_df["Count"].values,
            labels=uni_summ_df[feature].values,
            colors=style.color_gen(name=color_map, num=len(uni_summ_df[feature].values)),
            alpha=0.8,
            ax=ax,
        )

        # add canvas to prettierplot object
        ax = p.make_canvas(title="Category counts by target\n* {}".format(feature), position=132)

        # add faceted categorical plot to canvas
        p.facet_cat(
            df=bi_summ_df,
            feature=feature,
            label_rotate=rotation,
            color_map=color_map,
            bbox=(1.0, 1.15),
            alpha=0.8,
            legend_labels=legend_labels,
            x_units=None,
            ax=ax,
        )

        # add canvas to prettierplot object
        ax = p.make_canvas(title="Target proportion by category\n* {}".format(feature), position=133)

        # add stacked bar chart to canvas
        p.stacked_bar_h(
            df=prop_df.drop("Class", axis=1),
            bbox=(1.0, 1.15),
            legend_labels=legend_labels,
            color_map=color_map,
            alpha=0.8,
            ax=ax,
        )

        plt.show()


def eda_cat_target_num_feat(self, feature, color_map="viridis", outliers_out_of_scope=None, legend_labels=None,
                            chart_scale=15):
    """
    Documentation:

        ---
        Description:
            Creates exploratory data visualizations and statistical summaries for a number
            feature in the context of a categorical target.

        ---
        Parameters:
            feature : string
                Feature to visualize.
            color_map : string specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            outliers_out_of_scope : boolean, float or int, default=None
                Truncates the x-axis upper limit so that outliers are out of scope of the visualization.
                The x-axis upper limit is reset to the maximum non-outlier value.

                To identify outliers, the IQR is calculated, and values that are below the first quartile
                minus the IQR, or above the third quarterile plus the IQR are designated as outliers. If True
                is passed as a value, the IQR that is subtracted/added is multiplied by 5. If a float or int is
                passed, the IQR is multiplied by that value. Higher values increase how extremem values need
                to be to be identified as outliers.
            legend_labels : list, default=None
                Class labels displayed in plot legend.
            chart_scale : int or float, default=15
                Controls size and proportions of chart and chart elements. Higher value creates larger plots
                and increases visual elements proportionally.
    """
    ### data summaries
    ## bivariate roll_up table
    # combine feature column and target
    bi_df = pd.concat([self.data[feature], self.target], axis=1)

    # remove any rows with nulls
    bi_df = bi_df[bi_df[feature].notnull()]

    # bivariate summary statistics
    bi_summ_stats_df = pd.DataFrame(
        columns=["Class", "Count", "Proportion", "Mean", "StdDev"]
    )

    # for each unique class label
    for labl in np.unique(self.target):

        # get feature values associated with single class label
        feature_slice = bi_df[bi_df[self.target.name] == labl][feature]

        # append summary statistics for feature values associated with class label
        bi_summ_stats_df = bi_summ_stats_df.append(
            {
                "Class": labl,
                "Count": len(feature_slice),
                "Proportion": len(feature_slice) / len(bi_df[feature]) * 100,
                "Mean": np.mean(feature_slice),
                "StdDev": np.std(feature_slice),
            },
            ignore_index=True,
        )

    # apply custom legend labels, or set dtype to int if column values are numeric
    if legend_labels is not None:
        bi_summ_stats_df["Class"] = legend_labels
    elif is_numeric_dtype(bi_summ_stats_df["Class"]):
        bi_summ_stats_df["Class"] = bi_summ_stats_df["Class"].astype(np.int)


    ## Feature summary
    describe_df = pd.DataFrame(bi_df[feature].describe()).reset_index()

    # add missing percentage
    describe_df = describe_df.append(
        {
            "index": "missing",
            feature: np.round(self.data.shape[0] - bi_df[feature].shape[0], 5),
        },
        ignore_index=True,
    )

    # add skew
    describe_df = describe_df.append(
        {
            "index": "skew",
            feature: np.round(stats.skew(bi_df[feature].values, nan_policy="omit"), 5),
        },
        ignore_index=True,
    )
    # add kurtosis
    describe_df = describe_df.append(
        {
            "index": "kurtosis",
            feature: stats.kurtosis(bi_df[feature].values, nan_policy="omit"),
        },
        ignore_index=True,
    )
    describe_df = describe_df.rename(columns={"index": ""})

    # execute z-test or t-test
    if len(np.unique(self.target)) == 2:
        s1 = bi_df[
            (bi_df[self.target.name] == bi_df[self.target.name].unique()[0])
        ][feature]
        s2 = bi_df[
            (bi_df[self.target.name] == bi_df[self.target.name].unique()[1])
        ][feature]
        if len(s1) > 30 and len(s2) > 30:

            # perform z-test, return z-statistic and p-value
            z, p_val = ztest(s1, s2)

            # add z-statistic and p-value to DataFrame
            stat_test_df = pd.DataFrame(
                data=[{"z-test statistic": z, "p-value": p_val}],
                columns=["z-test statistic", "p-value"],
                index=[feature],
            ).round(4)
        else:
            # perform t-test, return t-score and p-value
            t, p_val = stats.ttest_ind(s1, s2)

            # add t-statistic and p-value to DataFrame
            stat_test_df = pd.DataFrame(
                data=[{"t-test statistic": t, "p-value": p_val}],
                columns=["t-test statistic", "p-value"],
                index=[feature],
            ).round(4)

        # display summary tables
        self.df_side_by_side(
            dfs=(describe_df, bi_summ_stats_df, stat_test_df),
            names=["Feature summary", "Feature vs. target summary", "Statistical test"],
        )
    else:

        # display summary tables
        self.df_side_by_side(
            dfs=(describe_df, bi_summ_stats_df),
            names=["Feature summary", "Feature vs. target summary"],
        )

    ### visualizations
    # create prettierplot object
    p = PrettierPlot(chart_scale=chart_scale, plot_orientation="wide_standard")

    # if boolean is passed to outliers_out_of_scope
    if isinstance(outliers_out_of_scope, bool):
        # if outliers_out_of_scope = True
        if outliers_out_of_scope:

            # identify outliers using IQR method and an IQR step of 5
            outliers = self.outlier_IQR(self.data[feature], iqr_step=5)

            # reset x-axis minimum and maximum
            x_axis_min = self.data[feature].drop(index=outliers).min()
            x_axis_max = self.data[feature].drop(index=outliers).max()
    # if outliers_out_of_scope is a float or int
    elif isinstance(outliers_out_of_scope, float) or isinstance(outliers_out_of_scope, int):
        # identify outliers using IQR method and an IQR step equal to the float/int passed
        outliers = self.outlier_IQR(self.data[feature], iqr_step=outliers_out_of_scope)

        # reset x-axis minimum and maximum
        x_axis_min = self.data[feature].drop(index=outliers).min()
        x_axis_max = self.data[feature].drop(index=outliers).max()

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title="Feature distribution\n* {}".format(feature),
        title_scale=0.85,
        position=221,
    )

    ## dynamically determine precision of x-units
    # capture min and max feature values
    dist_min = bi_df[feature].values.min()
    dist_max = bi_df[feature].values.max()

    # determine x-units precision based on min and max values in feature
    if -3 < dist_min < 3 and -3 < dist_max < 3 and dist_max/dist_min < 10:
        x_units = "fff"
    elif -30 < dist_min < 30 and -30 < dist_max < 30 and dist_max/dist_min < 3:
        x_units = "fff"
    elif -5 < dist_min < 5 and -5 < dist_max < 5 and dist_max/dist_min < 10:
        x_units = "ff"
    elif -90 < dist_min < 90 and -90 < dist_max < 90 and dist_max/dist_min < 5:
        x_units = "ff"
    else:
        x_units = "f"

    # add distribution plot to canvas
    p.dist_plot(
        bi_df[feature].values,
        color=style.style_grey,
        y_units="f",
        x_units=x_units,
        ax=ax,
    )

    # optionally reset x-axis limits
    if outliers_out_of_scope is not None:
        plt.xlim(x_axis_min, x_axis_max)

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title="Probability plot\n* {}".format(feature),
        title_scale=0.85,
        position=222,
    )

    # add QQ / probability plot to canvas
    p.prob_plot(
        x=bi_df[feature].values,
        plot=ax,
    )

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title="Distribution by class\n* {}".format(feature),
        title_scale=0.85,
        position=223,
    )

    ## dynamically determine precision of x-units
    # capture min and max feature values
    dist_min = bi_df[feature].values.min()
    dist_max = bi_df[feature].values.max()

    # determine x-units precision based on min and max values in feature
    if -3 < dist_min < 3 and -3 < dist_max < 3 and dist_max/dist_min < 10:
        x_units = "fff"
    elif -30 < dist_min < 30 and -30 < dist_max < 30 and dist_max/dist_min < 3:
        x_units = "fff"
    elif -5 < dist_min < 5 and -5 < dist_max < 5 and dist_max/dist_min < 10:
        x_units = "ff"
    elif -90 < dist_min < 90 and -90 < dist_max < 90 and dist_max/dist_min < 5:
        x_units = "ff"
    else:
        x_units = "f"

    # generate color list
    color_list = style.color_gen(name=color_map, num=len(np.unique(self.target)))

    # add one distribution plot to canvas for each category class
    for ix, labl in enumerate(np.unique(bi_df[self.target.name].values)):
        p.dist_plot(
            bi_df[bi_df[self.target.name] == labl][feature].values,
            color=color_list[ix],
            y_units="f",
            x_units=x_units,
            legend_labels=legend_labels if legend_labels is not None else np.arange(len(np.unique(self.target))),
            alpha=0.4,
            bbox=(1.0, 1.0),
            ax=ax,
        )

    # optionally reset x-axis limits
    if outliers_out_of_scope is not None:
        plt.xlim(x_axis_min, x_axis_max)

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title="Boxplot by class\n* {}".format(feature),
        title_scale=0.85,
        position=224,
    )

    ## dynamically determine precision of x-units
    # capture min and max feature values
    dist_min = bi_df[feature].values.min()
    dist_max = bi_df[feature].values.max()

    # determine x-units precision based on min and max values in feature
    if -3 < dist_min < 3 and -3 < dist_max < 3 and dist_max/dist_min < 10:
        x_units = "fff"
    elif -30 < dist_min < 30 and -30 < dist_max < 30 and dist_max/dist_min < 3:
        x_units = "fff"
    elif -5 < dist_min < 5 and -5 < dist_max < 5 and dist_max/dist_min < 10:
        x_units = "ff"
    elif -90 < dist_min < 90 and -90 < dist_max < 90 and dist_max/dist_min < 5:
        x_units = "ff"
    else:
        x_units = "f"

    # add horizontal box plot to canvas
    p.box_plot_h(
        x=feature,
        y=self.target.name,
        data=bi_df,
        alpha=0.7,
        x_units=x_units,
        legend_labels=legend_labels,
        bbox=(1.2, 1.0),
        suppress_outliers=True,
        ax=ax
        )

    # optionally reset x-axis limits
    if outliers_out_of_scope is not None:
        plt.xlim(x_axis_min-(x_axis_min * 0.1), x_axis_max)

    # apply position adjustment to subplots
    plt.subplots_adjust(bottom=-0.1)

    plt.show()


def eda_num_target_num_feat(self, feature, color_map="viridis", chart_scale=15):
    """
    Documentation:

        ---
        Description:
            Produces exploratory data visualizations and statistical summaries for a numeric
            feature in the context of a numeric target.

        ---
        Parameters:
            feature : string
                Feature to visualize.
            color_map : string specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            chart_scale : int or float, default=15
                Controls size and proportions of chart and chart elements. Higher value creates
                larger plots and increases visual elements proportionally.
    """
    ### data summaries
    ## feature summary
    # combine feature column and target
    bi_df = pd.concat([self.data[feature], self.target], axis=1)

    # remove any rows with nulls
    bi_df = bi_df[bi_df[feature].notnull()]

    # cast target as float
    bi_df[self.target.name] = bi_df[self.target.name].astype(float)

    # create summary statistic table
    describe_df = pd.DataFrame(bi_df[feature].describe()).reset_index()

    # add skew and kurtosis to describe_df
    describe_df = describe_df.append(
        {
            "index": "skew",
            feature: stats.skew(bi_df[feature].values, nan_policy="omit"),
        },
        ignore_index=True,
    )
    describe_df = describe_df.append(
        {
            "index": "kurtosis",
            feature: stats.kurtosis(bi_df[feature].values, nan_policy="omit"),
        },
        ignore_index=True,
    )
    describe_df = describe_df.rename(columns={"index": ""})

    # display summary tables
    display(describe_df)

    ### visualizations
    # create prettierplot object
    p = PrettierPlot(chart_scale=chart_scale, plot_orientation="wide_narrow")

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title="Feature distribution\n* {}".format(feature), position=131, title_scale=1.2
    )

    # determine x-units precision based on magnitude of max value
    if -1 <= np.nanmax(bi_df[feature].values) <= 1:
        x_units = "fff"
    elif -10 <= np.nanmax(bi_df[feature].values) <= 10:
        x_units = "ff"
    else:
        x_units = "f"

    # determine y-units precision based on magnitude of max value
    if -1 <= np.nanmax(bi_df[feature].values) <= 1:
        y_units = "fff"
    elif -10 <= np.nanmax(bi_df[feature].values) <= 10:
        y_units = "ff"
    else:
        y_units = "f"

    # x rotation
    if -10000 < np.nanmax(bi_df[feature].values) < 10000:
        x_rotate = 0
    else:
        x_rotate = 45

    # add distribution plot to canvas
    p.dist_plot(
        bi_df[feature].values,
        color=style.style_grey,
        y_units=y_units,
        x_rotate=x_rotate,
        ax=ax,
    )

    # add canvas to prettierplot object
    ax = p.make_canvas(title="Probability plot\n* {}".format(feature), position=132)

    # add QQ / probability plot to canvas
    p.prob_plot(x=bi_df[feature].values, plot=ax)

    # add canvas to prettierplot object
    ax = p.make_canvas(
        title="Regression plot - feature vs. target\n* {}".format(feature),
        position=133,
        title_scale=1.5
        )

    # add regression plot to canvas
    p.reg_plot(
        x=feature,
        y=self.target.name,
        data=bi_df,
        x_jitter=0.1,
        x_rotate=x_rotate,
        x_units=x_units,
        y_units=y_units,
        ax=ax,
    )
    plt.show()


def eda_num_target_cat_feat(self, feature, level_count_cap=50, color_map="viridis", chart_scale=15):
    """
    Documentation:

        ---
        Description:
            Produces exploratory data visualizations and statistical summaries for a category
            feature in the context of a numeric target.

        ---
        Parameters:
            feature : string
                Feature to visualize.
            level_count_cap : int, default=50
                Maximum number of unique levels in feature. If the number of levels exceeds the
                cap then the feature is skipped.
            color_map : string specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            chart_scale : int or float, default=15
                Controls size and proportions of chart and chart elements. Higher value creates
                larger plots and increases visual elements proportionally.
    """
    # if number of unique levels in feature is less than specified level_count_cap
    if (len(np.unique(self.data[self.data[feature].notnull()][feature].values)) < level_count_cap):

        ### data summaries
        ## feature summary
        # create empty DataFrame
        uni_summ_df = pd.DataFrame(columns=[feature, "Count", "Proportion"])

        # capture unique values and count of those unique values
        unique_vals, unique_counts = np.unique(
            self.data[self.data[feature].notnull()][feature], return_counts=True
        )

        # append each unique value, count and proportion to DataFrame
        for i, j in zip(unique_vals, unique_counts):
            uni_summ_df = uni_summ_df.append(
                {feature: i, "Count": j, "Proportion": j / np.sum(unique_counts) * 100},
                ignore_index=True,
            )

        # sort DataFrame by "Proportion", descending
        uni_summ_df = uni_summ_df.sort_values(by=["Proportion"], ascending=False)

        # set values to int dtype where applicable to optimize
        if is_numeric_dtype(uni_summ_df[feature]):
            uni_summ_df[feature] = uni_summ_df[feature].astype("int64")
        uni_summ_df["Count"] = uni_summ_df["Count"].astype("int64")

        ## feature vs. target summary
        # combine feature column and target
        bi_df = pd.concat([self.data[feature], self.target], axis=1)

        # remove any rows with nulls
        bi_df = bi_df[bi_df[feature].notnull()]

        # cast target as float
        bi_df[self.target.name] = bi_df[self.target.name].astype(float)

        # create pivot table of target summary statistics, grouping by category feature
        bi_summ_piv_df = pd.pivot_table(
            bi_df, index=feature, aggfunc={self.target.name: [np.nanmin, np.nanmax, np.nanmean, np.nanmedian, np.nanstd]}
        )
        multi_index = bi_summ_piv_df.columns
        single_index = pd.Index([i[1] for i in multi_index.tolist()])
        bi_summ_piv_df.columns = single_index
        bi_summ_piv_df.reset_index(inplace=True)
        bi_summ_piv_df = bi_summ_piv_df.rename(columns={
                                                "nanmin":"Min",
                                                "nanmax":"Max",
                                                "nanmean":"Mean",
                                                "nanmedian":"Median",
                                                "nanstd":"StdDev",
                                            }
                                        )
        # fill nan's with zero
        fill_columns = bi_summ_piv_df.iloc[:,1:].columns
        bi_summ_piv_df[fill_columns] = bi_summ_piv_df[fill_columns].fillna(0)

        # reorder column
        bi_summ_piv_df = bi_summ_piv_df[[feature,"Mean","Median","StdDev","Min","Max"]]

        # convert to int
        if is_numeric_dtype(bi_summ_piv_df[feature]):
            bi_summ_piv_df[feature] = bi_summ_piv_df[feature].astype("int64")

        # display summary tables
        self.df_side_by_side(
            dfs=(uni_summ_df, bi_summ_piv_df),
            names=["Feature summary", "Feature vs. target summary"],
        )

        ### visualizations
        # create prettierplot object
        p = PrettierPlot(chart_scale=chart_scale, plot_orientation="wide_narrow")

        # add canvas to prettierplot object
        ax = p.make_canvas(title="Category counts\n* {}".format(feature), position=131, title_scale=1.0)

        # add treemap to canvas
        p.tree_map(
            counts=uni_summ_df["Count"].values,
            labels=uni_summ_df[feature].values,
            colors=style.color_gen(name=color_map, num=len(uni_summ_df[feature].values)),
            alpha=0.8,
            ax=ax,
        )

        # add canvas to prettierplot object
        ax = p.make_canvas(title="Feature distribution\n* {}".format(feature), position=132)

        # error catching block for resorting labels
        try:
            sorted(unique_vals, key=int)
        except ValueError:
            pass
        else:
            # sort unique_vals/unique_counts for bar chart
            new_ix = [
                sorted(list(unique_vals), key=int).index(i) for i in list(unique_vals)
            ]
            unique_vals = np.array(sorted(list(unique_vals), key=int))
            unique_counts = np.array([y for x, y in sorted(zip(new_ix, unique_counts))])

            # sort temporary data frame for box plot
            bi_df[feature] = bi_df[feature].astype(int)

        # dynamically set rotation angle based on number unique values and maximum length of
        # category labels.
        len_unique_val = len(unique_vals)
        avg_len_unique_val = sum(map(len, str(unique_vals))) / len(unique_vals)
        if len_unique_val <= 4 and avg_len_unique_val <= 12:
            rotation = 0
        elif len_unique_val >= 5 and len_unique_val <= 8 and avg_len_unique_val <= 7.0:
            rotation = 0
        elif len_unique_val >= 9 and len_unique_val <= 14 and avg_len_unique_val <= 6:
            rotation = 0
        else:
            rotation = 30

        # represent x-axis tick labels as integers rather than floats
        x_values = list(map(str, unique_vals.tolist()))
        try:
            x_values = [int(float(x)) for x in x_values]
        except ValueError:
            pass

        # add bar chart to canvas
        p.bar_v(
            x=x_values,
            counts=unique_counts,
            label_rotate=rotation,
            color=style.style_grey,
            y_units="f",
            x_tick_wrap=True,
            ax=ax,
        )

        # hide every other label if total number of levels is greater than 40
        if len_unique_val > 40:
            n = 2
            [
                l.set_visible(False)
                for (i, l) in enumerate(ax.xaxis.get_ticklabels())
                if i % n != 0
            ]

        # add canvas to prettierplot object
        ax = p.make_canvas(
            title="Boxplot by category\n* {}".format(feature), position=133
        )

        ## dynamically determine precision of y-units
        # capture min and max feature values
        dist_min = bi_df[self.target.name].values.min()
        dist_max = bi_df[self.target.name].values.max()

        # determine y-units precision based on min and max values in feature
        if -3 < dist_min < 3 and -3 < dist_max < 3 and dist_max/dist_min < 10:
            y_units = "fff"
        elif -30 < dist_min < 30 and -30 < dist_max < 30 and dist_max/dist_min < 3:
            y_units = "fff"
        elif -5 < dist_min < 5 and -5 < dist_max < 5 and dist_max/dist_min < 10:
            y_units = "ff"
        elif -90 < dist_min < 90 and -90 < dist_max < 90 and dist_max/dist_min < 5:
            y_units = "ff"
        else:
            y_units = "f"

        # add vertical box plot to canvas
        p.box_plot_v(
            x=feature,
            y=self.target.name,
            data=bi_df.sort_values([feature]),
            color=matplotlib.cm.get_cmap(name=color_map),
            label_rotate=rotation,
            y_units=y_units,
            ax=ax,
        )

        # hide every other label if total number of levels is greater than 40
        if len_unique_val > 40:
            n = 2
            [
                l.set_visible(False)
                for (i, l) in enumerate(ax.xaxis.get_ticklabels())
                if i % n != 0
            ]

        plt.show()


def df_side_by_side(self, dfs, names=[]):
    """
    Documentation:

        ---
        Description:
            Helper function for displaying Pandas DataFrames side by side in a
            Jupyter Notebook.

        ---
        Parameters:
            dfs : list
                List of Pandas DataFrames to display.
            names : list, default=[]
                List of names to be displayed above Pandas DataFrames.
    """
    html_str = ""
    if names:
        html_str += (
            "<tr>"
            + "".join(f'<td style="text_align:center">{name}</td>' for name in names)
            + "</tr>"
        )
    html_str += (
        "<tr>"
        + "".join(
            f'<td style="vertical_align:top"> {df.to_html(index=False)}</td>'
            for df in dfs
        )
        + "</tr>"
    )
    html_str = f"<table>{html_str}</table>"
    html_str = html_str.replace("table", 'table style="display:inline"')
    display_html(html_str, raw=True)