import os
import sys
import importlib
import pandas as pd

import sklearn.model_selection as model_selection

class Machine:
    """
    Info:
        Description:
            Machine facilitates machine learning tasks spanning a typicaly end-to-end
            machine learning pipeline, including data cleaning, feature encoding, exploratory
            data analysis, data prepation, model building, model tuning and model evaluation.
    """

    # Import mlmachine submodules
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
        cleanLabel,
        ContextImputer,
        ConvertToCategory,
        DataFrameSelector,
        dataRefresh,
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
        featureSelectorCorr,
        featureSelectorCrossVal,
        featureSelectorFScoreClass,
        featureSelectorFScoreReg,
        featureSelectorImportance,
        featureSelectorResultsPlot,
        featureSelectorRFE,
        featureSelectorVariance,
        featureSelectorSummary,
        FeatureSync,
        featuresUsedSummary,
    )
    from .features.transform import (
        CustomBinner,
        EqualWidthBinner,
        featureDropper,
        NumericCoercer,
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
                    Preidentified categorical features that would otherwise be labeled as continuous.
                overrideNum : list, default = None
                    Preidentified continuous features that would otherwise be labeled as categorical.
                dateFeatures : list, default = None
                    Features comprised of calendar date values.
                target : list, default = None
                    Name of column containing dependent variable.
                targetType : list, default = None
                    Target variable type, either 'categorical' or 'continuous'
            Attributes:
                data : Pandas DataFrame
                    Independent variables returned as a Pandas DataFrame
                target : Pandas Series
                    Dependent variable returned as a Pandas Series
                featuresByDtype_ : dict
                    Dictionary contains keys 'continuous', 'categorical' and/or 'date'. The corresponding values
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

        # execute method measLevel
        if self.target is not None:
            self.data, self.target, self.featureByDtype = self.measLevel()
        else:
            self.data, self.featureByDtype = self.measLevel()


    def measLevel(self):
        """
        Documentation:
            Description:
                Determine level of measurement for each feature as categorical, continuous or date.
                Isolate dependent variable 'target', if provided, and drop from 'data' object.
        """
        ### identify target from features
        if self.target is not None:
            self.cleanLabel()

        ### add categorical and continuous keys, and any associated overrides
        self.featureByDtype = {}

        # categorical
        if self.overrideCat is None:
            self.featureByDtype["categorical"] = []
        else:
            self.featureByDtype["categorical"] = self.overrideCat

        # continuous
        if self.overrideNum is None:
            self.featureByDtype["continuous"] = []
        else:
            self.featureByDtype["continuous"] = self.overrideNum

        # date
        if self.dateFeatures is None:
            self.featureByDtype["date"] = []
        else:
            self.featureByDtype["date"] = self.dateFeatures

        # combined dictionary values for later filtering
        handled = [i for i in sum(self.featureByDtype.values(), [])]

        ### categorize remaining columns
        for c in [i for i in self.data.columns if i not in handled]:

            # identify feature type based on column data type
            if str(self.data[c].dtype).startswith(("int", "float")):
                self.featureByDtype["continuous"].append(c)
            elif str(self.data[c].dtype).startswith(("object")):
                self.featureByDtype["categorical"].append(c)

        ### return objects
        if self.target is not None:
            return self.data, self.target, self.featureByDtype
        else:
            return self.data, self.featureByDtype


    def edaData(self, data=None, target=None):
        """
        Documentation:
            Description:
                Simple helper function for mergeing together the 'data' variable and
                'target' variable. Intended to be used primarily when when it makes
                sense to pass the full combined dataset into a data visualization
                function.
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

    def featureByDtypeUpdate(self, data=None, featureByDtype=None, override=None):
        """
        Documentation:
            Description:
                Update featureByDtype dictionary with columns added during preprocessing. If necessary,
                change data type for object columns to numeric.
            Parameters:
                data : Pandas DataFrame, default = None
                    Pandas DataFrame containing independent variables. If left as None,
                    the feature dataset provided to Machine during instantiation is used.
                featureByDtype : dictionary, default = None
                    Dictionary containing string/list key/value pairs, where the key is a string that is either
                    "continuous" or "categorical", and the value is a list of string corresponding to columns.
                    If left as None, the featureByDtype dictionary created during instantiation is used.
            Return:
                data : Pandas DataFrame
                    Pandas DataFrame containing combined independent and dependent variables.
                featureByDtype : dictioanry
                    Pandas DataFrame containing combined independent and dependent variables.
        """
        # use data/featureByDtype["continuous"] columns provided during instantiation if left unspecified
        if data is None:
            data = self.data
        if featureByDtype is None:
            featureByDtype = self.featureByDtype

        # if an override dictionary is provided, utilize it first
        if override is not None:
            for columnType in override.keys():
                for column in override[columnType]:
                    featureByDtype[columnType].append(column)

        ## numeric column updates
        # capture numeric columns not currently in featureByDtype
        untrackedNumCols = list(set(data.select_dtypes(include=['number']).columns) - set(sum(featureByDtype.values(), [])))

        # append to featureByDtype list
        for numCol in untrackedNumCols:
            featureByDtype["continuous"].append(numCol)

        ## categorical column updates
        # capture object columns not currently in featureByDtype
        untrackedObjCols = list(set(data.select_dtypes(exclude=['number']).columns) - set(sum(featureByDtype.values(), [])))

        for objCol in untrackedObjCols:
            # test if the object column can be successfully converted to numeric
            # and if it is, then add it to the continuous key in featureByDtype
            try:
                data[objCol] = data[objCol].apply(pd.to_numeric)

                # append to featureByDtype list
                featureByDtype["continuous"].append(objCol)

            except ValueError:
                # append to featureByDtype list
                featureByDtype["categorical"].append(objCol)
        return data, featureByDtype


def trainTestCompile(data, targetCol, testSize = 0.2, randomState = 1):
    """
    Description:
        Intakes a dataset and returns
    Parameters:
        data: Pandas DataFrame or array
            Dataset to be deconstructed into train and test sets.
        targetCol : string
            Name of target column in data parameter
        testSize : float, default = 0.2
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
            X, y, test_size=0.2, random_state=1, stratify=y
        )

    dfTrain = XTrain.merge(yTrain, left_index=True, right_index=True)
    dfValid = XValid.merge(yValid, left_index=True, right_index=True)

    return dfTrain, dfValid