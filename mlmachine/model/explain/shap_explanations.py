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
        # "SVC",
        "SVR",
        # "XGBClassifier",
        # "XGBRegressor",
        # "LGBMClassifier",
        # "LGBMRegressor",
        # "RandomForestClassifier",
        # "GradientBoostingClassifier",
        # "AdaBoostClassifier", # not include in this explainer
        # "ExtraTreesClassifier",
        # "IsolationForest",
        # "RandomForestRegressor",
        # "GradientBoostingRegressor",
        # "ExtraTreesRegressor",
        # "AdaBoostRegressor", # not include in this explainer
    ],
}

def create_shap_explainers(self, model_dir):
    """

    """
    for model_file in os.listdir(model_dir)[1:]:

        #
        with open(os.path.join(model_dir, model_file), "rb") as handle:
            model = pickle.load(handle)
        estimator_name = model.__class__.__name__

        for explainer_name, valid_estimators in explainer_algorithm_map.items():
            if estimator_name in valid_estimators:

                explainer = eval(explainer_name)

                if explainer_name == "KernelExplainer":
                    explainer = explainer(model.predict_proba, self.data.values)
                else:
                    explainer = explainer(model, self.data.values)
                    shap_values = explainer(self.data.values)

                with open(os.path.join(os.path.split(model_dir)[0], "shap_explainers", "{}_{}.pkl".format(estimator_name, explainer_name)), 'wb') as handle:
                    pickle.dump(explainer, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print(estimator_name)
                print(explainer_name)
                print()

