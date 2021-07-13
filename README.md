[![PyPI version](https://badge.fury.io/py/mlmachine.svg)](https://badge.fury.io/py/mlmachine)

# mlmachine

<i>"mlmachine is a Python library that organizes and accelerates notebook-based machine learning experiments."</i>

## Table of Contents

- [Novel Functionality](#Novel-Functionality)
- [Example Notebooks](#Example-Notebooks)
- [Articles on Medium](#Articles-on-Medium)
- [Installation](#Installation)
- [Feedback](#Feedback)
- [Acknowledgments](#Acknowledgments)


## Novel Functionality

__Easy, Elegant EDA__

mlmachine creates beautiful and informative EDA panels with ease:

```python
# create EDA panel for all "category" features
for feature in mlmachine_titanic.data.mlm_dtypes["category"]:
    mlmachine_titanic.eda_cat_target_cat_feat(
        feature=feature,
        legend_labels=["Died","Survived"],
    )
```
![alt text](/notebooks/images/eda_loop.gif "EDA loop")
<br><br>



__Pandas-in / Pandas-out Pipelines__

mlmachine makes Scikit-learn transformers Pandas-friendly.

Here's an example. See how simply wrapping the mlmachine utility `PandasTransformer()` around `OneHotEncoder()` maintains our `DataFrame`:

![alt text](/notebooks/images/p1_pandastransformer.jpeg "Pandas Pipeline")
<br><br>



__KFold Target Encoding__

mlmachine includes a utility called `KFoldEncoder`, which applies target encoding on categorical features and leverages out-of-fold encoding to prevent target leakage:

```python
# perform 5-fold target encoding with TargetEncoder from the category_encoders library
encoder = KFoldEncoder(
    target=mlmachine_titanic_machine.training_target,
    cv=KFold(n_splits=5, shuffle=True, random_state=0),
    encoder=TargetEncoder,
)
encoder.fit_transform(mlmachine_titanic_machine.training_features[["Pclass"]])
```

![alt text](/notebooks/images/kfold.jpeg "Pandas Pipeline")
<br><br>



__Crowd-sourced Feature Importance & Exhaustive Feature Selection__

mlmachine employs a robust approach to estimating feature importance by using a variety of techniques:

- Tree-based Feature Importance
- Recursive Feature Elimination
- Sequential Forward Selection
- Sequential Backward Selection
- F-value / p-value
- Variance 
- Target Correlation

This occurs with one simple execution, and operates on multiple estimators and/or models, and one or more scoring metrics:

```python
# instantiate custom models
rf2 = RandomForestClassifier(max_depth=2)
rf4 = RandomForestClassifier(max_depth=4)
rf6 = RandomForestClassifier(max_depth=6)

# estimator list - default XGBClassifier, default
# RandomForestClassifier and three custom models
estimators = [
    XGBClassifier,
    RandomForestClassifier,
    rf2,
    rf4,
    rf6,
]

# instantiate FeatureSelector object
fs = mlmachine_titanic_machine.FeatureSelector(
    data=mlmachine_titanic_machine.training_features,
    target=mlmachine_titanic_machine.training_target,
    estimators=estimators,
)

# run feature importance techniques, use ROC AUC and
# accuracy score metrics and 0 CV folds (where applicable)
feature_selector_summary = fs.feature_selector_suite(
    sequential_scoring=["roc_auc","accuracy_score"],
    sequential_n_folds=0,
    save_to_csv=True,
)
```

Then the features are winnowed away, from least important to most important, through an exhaustive cross-validation procedure in search of an optimum feature subset:

![alt text](/notebooks/images/feature_selection.jpg "Pandas Pipeline")


<br><br>



__Hyperparameter Tuning with Bayesian Optimization__

mlmachine can perform Bayesian optimization on multiple estimators in one shot, and includes functionality for visualizing model performance and parameter selections:

```python
# generate parameter selection panels for each parameter
mlmachine_titanic_machine.model_param_plot(
        bayes_optim_summary=bayes_optim_summary,
        estimator_class="KNeighborsClassifier",
        estimator_parameter_space=estimator_parameter_space,
        n_iter=100,
    )
```
![alt text](/notebooks/images/param_loop.gif "EDA loop")
<br><br>




## Example Notebooks

All examples can be viewed [here](https://github.com/petersontylerd/mlmachine/tree/master/notebooks)

[Example Notebook 1](https://github.com/petersontylerd/mlmachine/tree/master/notebooks/mlmachine_part_1.ipynb) - Learn the basics of mlmachine, how to create EDA panels, and how to execute Pandas-friendly Scikit-learn transformations and pipelines.

[Example Notebook 2](https://github.com/petersontylerd/mlmachine/tree/master/notebooks/mlmachine_part_2.ipynb) - Learn how use mlmachine to assess a datasets pre-processing needs. See examples of how to use novel functionality, such as `GroupbyImputer()`, `KFoldEncoder()` and `DualTransformer()`.

[Example Notebook 3](https://github.com/petersontylerd/mlmachine/tree/master/notebooks/mlmachine_part_3.ipynb) - Learn how to perform thorough feature importance estimation, followed by an exhaustive, cross-validation-driven feature selection process.

[Example Notebook 4](https://github.com/petersontylerd/mlmachine/tree/master/notebooks/mlmachine_part_4.ipynb) - Learn how to execute hyperparameter tuning with Bayesian optimization for multiple model and multiple parameter spaces in one simple execution.



## Articles on Medium

[mlmachine - Clean ML Experiments, Elegant EDA & Pandas Pipelines](https://towardsdatascience.com/mlmachine-clean-ml-experiments-elegant-eda-pandas-pipelines-daba951dde0a) - Published 4/3/2020

[mlmachine - GroupbyImputer, KFoldEncoder, and Skew Correction](https://towardsdatascience.com/mlmachine-groupbyimputer-kfoldencoder-and-skew-correction-357f202d2212) - Published 4/13/2020



## Installation

__Python Requirements__: 3.6, 3.7

mlmachine uses the latest, or almost latest, versions of all dependencies. Therefore, it is highly recommended that mlmachine is installed in a virtual environment.

_**pyenv**_

Create a new virtual environment:

`$ pyenv virtualenv 3.7.5 mlmachine-env`

Activate your new virtual environment:

`$ pyenv activate mlmachine-env`

Install mlmachine using pip to install mlmachine and all dependencies:

`$ pip install mlmachine`

_**anaconda**_

Create a new virtual environment:

`$ conda create --name mlmachine-env python=3.7`

Activate your new virtual environment:

`$ conda activate mlmachine-env`

Install mlmachine using pip to install mlmachine and all dependencies:

`$ pip install mlachine`

## Feedback

Any and all feedback is welcome. Please send me an email at petersontylerd@gmail.com

## Acknowledgments

mlmachine stands on the shoulders of many great Python packages:

[catboost](https://github.com/catboost/catboost) | [category_encoders](https://github.com/scikit-learn-contrib/categorical-encoding) | [eif](https://github.com/sahandha/eif) | [hyperopt](https://github.com/hyperopt/hyperopt) | [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) | [jupyter](https://github.com/jupyter/notebook) | [lightgbm](https://github.com/microsoft/LightGBM) | [matplotlib](https://github.com/matplotlib/matplotlib) | [numpy](https://github.com/numpy/numpy) | [pandas](https://github.com/pandas-dev/pandas) | [prettierplot](https://github.com/petersontylerd/prettierplot) | [scikit-learn](https://github.com/scikit-learn/scikit-learn) | [scipy](https://github.com/scipy/scipy) | [seaborn](https://github.com/mwaskom/seaborn) | [shap](https://github.com/slundberg/shap) | [statsmodels](https://github.com/statsmodels/statsmodels) | [xgboost](https://github.com/dmlc/xgboost) |
