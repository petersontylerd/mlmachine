import os
import pickle

import pandas as pd

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
        # "XGBClassifier",
        # "XGBRegressor",
        # "LGBMClassifier",
        # "LGBMRegressor",
        # "RandomForestClassifier",
        "GradientBoostingClassifier",
        # "ExtraTreesClassifier",
        # "RandomForestRegressor",
        # "GradientBoostingRegressor",
        # "ExtraTreesRegressor",
    ],
    "LinearExplainer": [
        # "Lasso",
        # "Ridge",
        # "ElasticNet",
        # "LinearRegression",
        # "LogisticRegression",
    ],
    "KernelExplainer": [
        # "KNeighborsClassifier",
        # "KNeighborsRegressor",
        # "SVC",
        # "SVR",
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

                # convert str into explainer class
                explainer = eval(explainer_name)

                # generate shap explainer and shap values with KernelExplainer
                if explainer_name in ["KernelExplainer"]:
                    # explainer = explainer(model.predict_proba, self.training_features.values)
                    # shap_values = explainer.shap_values(self.training_features.values)
                    continue

                # generate shap explainer and shap values with LinearExplainer
                elif explainer_name in ["LinearExplainer"]:
                    explainer = explainer(model, self.training_features.values)
                    training_shap_values = explainer.shap_values(self.training_features.values)
                    validation_shap_values = explainer.shap_values(self.validation_features.values)

                # generate shap explainer and shap values with TreeExplainer
                else:
                    explainer = explainer(model)
                    try:
                        training_shap_values = explainer.shap_values(self.training_features.values)
                        validation_shap_values = explainer.shap_values(self.validation_features.values)

                    # if the process errors due to an addivity error, run with check_additivity set to False
                    except Exception:
                        training_shap_values = explainer.shap_values(self.training_features.values, check_additivity=False)
                        validation_shap_values = explainer.shap_values(self.validation_features.values, check_additivity=False)

                # capture SHAP values in a DataFrame and preserve the index

                try:
                    training_shap_values = pd.DataFrame(
                                        training_shap_values[0],
                                        index=self.training_target.index,
                                        columns=self.training_features.columns,
                                    )
                    validation_shap_values = pd.DataFrame(
                                        validation_shap_values[0],
                                        index=self.validation_target.index,
                                        columns=self.validation_features.columns,
                                    )
                except ValueError:
                    training_shap_values = pd.DataFrame(
                                        training_shap_values,
                                        index=self.training_target.index,
                                        columns=self.training_features.columns,
                                    )
                    validation_shap_values = pd.DataFrame(
                                        validation_shap_values,
                                        index=self.validation_target.index,
                                        columns=self.validation_features.columns,
                                    )

                # save shap explainer object as .pkl file
                with open(os.path.join(self.shap_explainers_object_dir, f"{estimator_name}_{explainer_name}.pkl"), 'wb') as handle:
                    pickle.dump(explainer, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # save shap values as .pkl file
                with open(os.path.join(self.shap_values_object_dir, f"{estimator_name}_{explainer_name}_training.pkl"), 'wb') as handle:
                    pickle.dump(training_shap_values, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.shap_values_object_dir, f"{estimator_name}_{explainer_name}_validation.pkl"), 'wb') as handle:
                    pickle.dump(validation_shap_values, handle, protocol=pickle.HIGHEST_PROTOCOL)

