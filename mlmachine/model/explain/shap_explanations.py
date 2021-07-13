import os
import pickle

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    IsolationForest,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import (
    Lasso,
    Ridge,
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    SGDRegressor,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.kernel_ridge import KernelRidge

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import catboost

from shap import (
    DeepExplainer,
    AdditiveExplainer,
    Explainer,
    GradientExplainer,
    TreeExplainer,
    KernelExplainer,
    LinearExplainer,
    PermutationExplainer,
    PartitionExplainer,
    SamplingExplainer,
)

explainer_algorithm_map = {
    # "Explainer": [
    #     ""
    # ],
    "TreeExplainer": [
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        # "AdaBoostClassifier", # not include in this explainer
        "ExtraTreesClassifier",
        "IsolationForest",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "ExtraTreesRegressor",
        # "AdaBoostRegressor", # not include in this explainer
    ],
    "LinearExplainer": [
        "Lasso",
        "Ridge",
        "ElasticNet",
        "LinearRegression",
        "LogisticRegression",
        "SGDRegressor",
    ],
    "KernelExplainer": [
        "KNeighborsClassifier",
        "KNeighborsRegressor",
        "SVC",
        "SVR",
    ],
}

def create_shap_explainers(self):
    """

    """

    # iterate through all .pkl files in the trained model directory
    for model_file in os.listdir(self.training_models_object_dir):

        # reload model from .pkl file
        with open(os.path.join(self.training_models_object_dir, model_file), "rb") as handle:
            model = pickle.load(handle)
        
        # capture estimator object name
        estimator_name = model.__class__.__name__

        # iteration through explainer type and its associated estimators
        for explainer_name, valid_estimators in explainer_algorithm_map.items():
            if estimator_name in valid_estimators:

                explainer = eval(explainer_name)
                print(estimator_name)
                print(explainer_name)
                print()

                # generate shap explainer and shap values with KernelExplainer
                if explainer_name in ["KernelExplainer"]:
                    explainer = explainer(model.predict_proba, self.training_features.iloc[:10,:].values)
                    shap_values = explainer.shap_values(self.training_features.iloc[:10,:].values)
                
                # generate shap explainer and shap values with LinearExplainer
                elif explainer_name in ["LinearExplainer"]:
                    explainer = explainer(model, self.training_features.iloc[:10,:].values)
                    shap_values = explainer.shap_values(self.training_features.iloc[:10,:].values)
                # generate shap explainer and shap values with TreeExplainer
                else:
                    explainer = explainer(model)
                    try:
                        shap_values = explainer.shap_values(self.training_features.iloc[:10,:].values)
                    
                    # if the process errors due to an addivity error, run with check_additivity set to False
                    except Exception:
                        shap_values = explainer.shap_values(self.training_features.iloc[:10,:].values, check_additivity=False)

                # save shap explainer object as .pkl file
                with open(os.path.join(self.shap_explainers_object_dir, f"{estimator_name}_{explainer_name}.pkl"), 'wb') as handle:
                    pickle.dump(explainer, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # save shap values as .pkl file
                with open(os.path.join(self.shap_values_object_dir, f"{estimator_name}_{explainer_name}.pkl"), 'wb') as handle:
                    pickle.dump(shap_values, handle, protocol=pickle.HIGHEST_PROTOCOL)


