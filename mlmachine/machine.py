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
        edaNumTargetNumFeat,
        edaCatTargetCatFeat,
        edaCatTargetNumFeat,
        edaNumTargetCatFeat,
        dfSideBySide,
    )
    from .explore.edaTransform import (
        edaTransformInitial,
        edaTransformLog1,
        edaTransformBoxCox,
    )
    from .features.encode import (
        cleanLabel,
        Dummies,
        MissingDummies,
        OrdinalEncoder,
        CustomOrdinalEncoder,
    )
    from .features.importance import (
        featureImportanceSummary,
    )
    from .features.impute import (
        NumericalImputer,
        ModeImputer,
        ConstantImputer,
        ContextImputer,
    )
    from .features.missing import (
        missingDataDropperAll,
        featureDropper,
        missingColCompare,
    )
    from .features.outlier import (
        OutlierIQR,
        ExtendedIsoForest,
        outlierSummary,
    )
    from .features.scale import (
        Standard,
        Robust,
    )
    from .features.transform import (
        skewSummary,
        SkewTransform,
        EqualWidthBinner,
        CustomBinner,
        PercentileBinner,
        featureDropper,
    )
    from .model.evaluate.summarize import (
        topBayesOptimModels,
    )    
    from .model.evaluate.visualize import (
        classificationPanel,
        singleShapValueTree,
        singleShapVizTree,
        multiShapValueTree,
        multiShapVizTree,
        shapDependencePlot,
        shapSummaryPlot,
    )    
    from .model.tune.bayesianOptimSearch import (
        BayesOptimModelBuilder,
        execBayesOptimSearch,
        objective,
        unpackBayesOptimSummary,
        modelLossPlot,
        modelParamPlot,
        samplePlot,
    )
    from .model.tune.powerGridSearch import (
        PowerGridSearcher,
        powerGridModelBuilder,
    )
    from .model.tune.stack import (
        oofGenerator,
        modelStacker,
    )
    
    def __init__(self, data, removeFeatures=[], overrideCat=None, overrideNum=None,
                dateFeatures=None, target=None, targetType=None):
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
            self.data, self.target, self.featureByDtype_ = self.measLevel()
        else:
            self.data, self.featureByDtype_ = self.measLevel()

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
        self.featureByDtype_ = {}

        # categorical
        if self.overrideCat is None:
            self.featureByDtype_["categorical"] = []
        else:
            self.featureByDtype_["categorical"] = self.overrideCat

        # continuous
        if self.overrideNum is None:
            self.featureByDtype_["continuous"] = []
        else:
            self.featureByDtype_["continuous"] = self.overrideNum

        # date
        if self.dateFeatures is None:
            self.featureByDtype_["date"] = []
        else:
            self.featureByDtype_["date"] = self.dateFeatures

        # combined dictionary values for later filtering
        handled = [i for i in sum(self.featureByDtype_.values(), [])]

        ### categorize remaining columns
        for c in [i for i in self.data.columns if i not in handled]:

            # identify feature type based on column data type
            if str(self.data[c].dtype).startswith(("int", "float")):
                self.featureByDtype_["continuous"].append(c)
            elif str(self.data[c].dtype).startswith(("object")):
                self.featureByDtype_["categorical"].append(c)

        ### return objects
        if self.target is not None:
            return self.data, self.target, self.featureByDtype_
        else:
            return self.data, self.featureByDtype_

    def edaData(self, X, y):
        """
        Documentation:
            Description:
                Simple helper function for mergeing together the 'data' variable and
                'target' variable. Intended to be used primarily when when it makes
                sense to pass the full combined dataset into a data visualization
                function.
            Parameters:
                X : Pandas DataFrame
                    Pandas DataFrame containing independent variables
                y : Pandas Series
                    Pandas Series containing dependent target variable.
            Return:
                x : Pandas DataFrame
                    Pandas DataFrame containing combined independent and dependent variables.
        """
        return X.merge(y, left_index=True, right_index=True)

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

        
        