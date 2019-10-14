import os
import sys
import importlib
import itertools
import numpy as np
import pandas as pd

import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing

class Machine:
    """
    Documentation:
        Description:
            Machine facilitates rapid machine learning experimentation tasks, including data
            cleaning, feature encoding, exploratory data analysis, data prepation, model building,
            model tuning and model evaluation.
    """

    # import mlmachine submodules
    from .explore.edaMissing import (
        edaMissingSummary,
    )
    from .explore.edaSuite import (
        dfSideBySide,
        edaCatTargetCatFeat,
        edaCatTargetNumFeat,
        edaNumTargetCatFeat,
        edaNumTargetNumFeat,
    )
    from .explore.edaTransform import (
        edaTransformBoxCox,
        edaTransformInitial,
        edaTransformLog1,
    )
    from .features.preprocessing import (
        ContextImputer,
        DataFrameSelector,
        DualTransformer,
        PandasFeatureUnion,
        PlayWithPandas,
        skewSummary,
        UnprocessedColumnAdder,
    )
    from .features.missing import (
        missingColCompare,
        missingDataDropperAll,
    )
    from .features.outlier import (
        ExtendedIsoForest,
        OutlierIQR,
        outlierSummary,
    )
    from .features.selection import (
        FeatureSelector,
        FeatureSync,
    )
    from .features.transform import (
        CustomBinner,
        EqualWidthBinner,
        PercentileBinner,
    )
    from .model.evaluate.summarize import (
        regressionResults,
        regressionStats,
        topBayesOptimModels,
    )
    from .model.evaluate.visualize import (
        classificationPanel,
        regressionPanel,
    )
    from .model.explain.visualize import (
        multiShapValueTree,
        multiShapVizTree,
        shapDependenceGrid,
        shapDependencePlot,
        shapSummaryPlot,
        singleShapValueTree,
        singleShapVizTree,
    )
    from .model.tune.bayesianOptimSearch import (
        BasicModelBuilder,
        BayesOptimModelBuilder,
        execBayesOptimSearch,
        modelLossPlot,
        modelParamPlot,
        objective,
        samplePlot,
        unpackBayesOptimSummary,
    )
    from .model.tune.powerGridSearch import (
        powerGridModelBuilder,
        PowerGridSearcher,
    )
    from .model.tune.stack import (
        modelStacker,
        oofGenerator,
    )


    def __init__(self, data, removeFeatures=[], overrideCat=None, overrideNum=None, dateFeatures=None, target=None,
                    targetType=None):
        """
        Documentation:
            Description:
                __init__ handles initial processing of main data set, identification of
                select features to be removed (if any), identification of select features
                to be considered as categorical despite the feature data type(s) (if any),
                identification of select features to be considered as numerical despite
                the feature data type(s) (if any), identification of select features to be
                considered as calendar date features (if any), identification of the feature
                representing the target (if there is one) and the type of target. Returns
                data frame of independent variables, series containing dependent variable
                and a dictionary that categorizes features by data type.
            Parameters:
                data : Pandas DataFrame
                    Input data provided as a Pandas DataFrame.
                removeFeatures : list, default = []
                    Features to be completely removed from dataset.
                overrideCat : list, default = None
                    Preidentified categorical features that would otherwise be labeled as numeric.
                overrideNum : list, default = None
                    Preidentified numeric features that would otherwise be labeled as categorical.
                dateFeatures : list, default = None
                    Features comprised of calendar date values.
                target : list, default = None
                    Name of column containing dependent variable.
                targetType : list, default = None
                    Target variable type, either 'categorical' or 'numeric'
            Attributes:
                data : Pandas DataFrame
                    Independent variables returned as a Pandas DataFrame
                target : Pandas Series
                    Dependent variable returned as a Pandas Series
                featuresByDtype_ : dict
                    Dictionary contains keys 'numeric', 'categorical' and/or 'date'. The corresponding values
                    are lists of column names that are of that feature type.
        """
        self.removeFeatures = removeFeatures
        self.target = data[target].squeeze() if target is not None else None
        self.data = (
            data.drop(self.removeFeatures + [self.target.name], axis=1)
            if target is not None
            else data.drop(self.removeFeatures, axis=1)
        )
        self.overrideCat = overrideCat
        self.overrideNum = overrideNum
        self.dateFeatures = dateFeatures
        self.targetType = targetType

        # execute method featureTypeCapture
        self.featureTypeCapture()

        # encode the target column if there is one
        if self.target is not None and self.targetType == "categorical":
            self.encodeTarget()

    def featureTypeCapture(self):
        """
        Documentation:
            Description:
                Determine feature type for each feature as being categorical or numeric.
        """
        ### populate featureType dictionary with feature type label for each each
        self.featureType = {}

        # categorical features
        if self.overrideCat is None:
            self.featureType["categorical"] = []
        else:
            self.featureType["categorical"] = self.overrideCat

            # convert column dtype to "category"
            self.data[self.overrideCat] = self.data[self.overrideCat].astype("category")

        # numeric features
        if self.overrideNum is None:
            self.featureType["numeric"] = []
        else:
            self.featureType["numeric"] = self.overrideNum

        # compile single list of features that have already been categorized
        handled = [i for i in sum(self.featureType.values(), [])]

        ### determine feature type for remaining columns
        for column in [i for i in self.data.columns if i not in handled]:
            # object columns set as categorical features
            if pd.api.types.is_object_dtype(self.data[column]):
                self.featureType["categorical"].append(column)

                # set column in dataset as categorical data type
                self.data[column] = self.data[column].astype("category")

            # numeric features
            elif pd.api.types.is_numeric_dtype(self.data[column]):
                self.featureType["numeric"].append(column)


    def featureTypeUpdate(self, columnsToDrop=None):
        """
        Documentation:
            Description:
                Update featureType dictionary to include new columns. Ensures new categorical columns
                in dataset have the dtype "category". Optionally drops specific columns from the dataset
                and featureType.
            Parameters:
                columnsToDrop : list, default = None
                    Columns to drop from output dataset(s)/
        """
        # set column data type as "category" where approrpirate
        for column in self.data.columns:
            if pd.api.types.is_object_dtype(self.data[column]):
                self.data[column] = self.data[column].astype("category")

        # determine columns already being tracked
        trackedColumns = []
        for k in self.featureType.keys():
            trackedColumns.append(self.featureType[k])
        trackedColumns = list(set(itertools.chain(*trackedColumns)))

        # remove any columns listed in columnsToDrop
        if columnsToDrop is not None:
            # remove from trackedColumns
            trackedColumns = [x for x in trackedColumns if x not in columnsToDrop]

            # remove from featureType
            for k in self.featureType.keys():
                self.featureType[k] = [x for x in self.featureType[k] if x not in columnsToDrop]

            # drop columns from main dataset
            self.data = self.data.drop(columnsToDrop, axis=1)

        # update featureType
        for column in [i for i in self.data.columns if i not in trackedColumns]:
            # categorical features
            if pd.api.types.is_categorical(self.data[column]):
                self.featureType["categorical"].append(column)
            # numeric features
            elif pd.api.types.is_numeric_dtype(self.data[column]):
                self.featureType["numeric"].append(column)

        # sort columns alphabeticalls by name
        self.data = self.data.sort_index(axis=1)


    def encodeTarget(self, reverse=False):
        """
        Documentation:
            Description:
                Encode categorical target column and store as a Pandas Series, where
                the name is the name of the feature in the original dataset.
            Parameters:
                reverse : boolean, default = False
                    Reverses encoding of target variables back to original variables.
        """
        # encode label
        self.le_ = preprocessing.LabelEncoder()

        # store as a named Pandas Series
        self.target = pd.Series(
            self.le_.fit_transform(self.target.values.reshape(-1)),
            name=self.target.name,
            index=self.target.index
        )

        print("******************\nCategorical label encoding\n")
        for origLbl, encLbl in zip(
            np.sort(self.le_.classes_), np.sort(np.unique(self.target))
        ):
            print("{} --> {}".format(origLbl, encLbl))

        # reverse the label encoding and overwrite target variable with the original data
        if reverse:
            self.target = self.le_.inverse_transform(self.target)

    def recombineData(self, data=None, target=None):
        """
        Documentation:
            Description:
                Helper function for recombining the features in the 'data' variable
                and the 'target' variable into one Pandas DataFrame.
            Parameters:
                data : Pandas DataFrame, default = None
                    Pandas DataFrame containing independent variables. If left as None,
                    the feature dataset provided to Machine during instantiation is used.
                target : Pandas Series, default = None
                    Pandas Series containing dependent target variable. If left as None,
                    the target dataset provided to Machine during instantiation is used.
            Return:
                df : Pandas DataFrame
                    Pandas DataFrame containing combined independent and dependent variables.
        """
        # use data/target provided during instantiation if left unspecified
        if data is None:
            data = self.data
        if target is None:
            target = self.target

        df = data.merge(target, left_index=True, right_index=True)
        return df


def trainTestCompile(data, targetCol, validSize = 0.2, randomState = 1):
    """
    Description:
        Intakes a single dataset and returns a training dataset and a validation dataset
        stored in Pandas DataFrames.
    Parameters:
        data: Pandas DataFrame or array
            Dataset to be deconstructed into train and test sets.
        targetCol : string
            Name of target column in data parameter
        validSize : float, default = 0.2
            Proportion of dataset to be set aside as "unseen" test data.
        randomState : int
            random number seed
    Returns:
        dfTrain : Pandas DataFrame
            Training dataset to be passed into mlmachine pipeline
        dfValid : Pandas DataFrame
            Validation dataset to be passed into mlmachine pipeline
    """
    if isinstance(data, pd.DataFrame):
        y = data[targetCol]
        X = data.drop([targetCol], axis=1)

    XTrain, XValid, yTrain, yValid = model_selection.train_test_split(
            X, y, test_size=validSize, random_state=1, stratify=y
        )

    dfTrain = XTrain.merge(yTrain, left_index=True, right_index=True)
    dfValid = XValid.merge(yValid, left_index=True, right_index=True)

    return dfTrain, dfValid