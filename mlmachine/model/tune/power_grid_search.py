from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np


class PowerGridSearcher:
    """
    documentation:
        definition:
            class capable of performing GridSearchCV and RandomizedSearchCV on
            multiple models, each with its own parameter grid, in a single execution.
            also returns a score summary sheet.
        parameters:
            models : dict
                dictionary of instantiated sklearn models.
            params : dict of dicts
                dictionary containing nested dictionaries of parameter grids for each model.
    """

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "some estimators are missing parameters: {0}".format(missing_params)
            )
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    # full GridSearchCV
    def fit_multi_gs_cv(
        self, x, y, cv=5, n_jobs=1, verbose=0, scoring=None, refit=True
    ):
        """
        documentation:
            definition:
                method for performing GridSearchCV on multiple models, each with its own parameter
                grid, in a single execution.
            parameters:
                x : array
                    input dataset.
                y : array
                    input dataset labels.
                cv : int, default = 5
                    the of folds to perform in cross_validation.
                verbose : int, default = 0
                    controls amount of information printed to console during fit.
                n_jobs : int, default = 1
                    number of works to deploy upon execution, if applicable.
                scoring : string (sklearn evaluation metric), default =None
                    number of works to deploy upon execution, if applicable.
                refit : boolean, default=True
                    dictates whether method is refit with the best model upon grid seach completion.
            returns:
                gs : GridSearchCV object
                    full GridSearchCV object.
        """
        for key in self.keys:
            print("running GridSearchCV for {0}".format(key))
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(
                model,
                params,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                scoring=scoring,
                refit=refit,
                return_train_score=True,
            )
            gs.fit(x, y)
            self.grid_searches[key] = gs
        return gs

    # RandomizedSearchCV
    def fit_multi_rgs_cv(
        self, x, y, cv=5, n_jobs=1, verbose=0, scoring=None, refit=True, n_iter=15
    ):
        """
        documentation:
            definition:
                method for performing randomized_GridSearchCV on multiple models, each with its own parameter
                grid, in a single execution.
            parameters:
                x : array
                    input dataset.
                y : array
                    input dataset labels.
                cv : int, default = 5
                    the of folds to perform in cross_validation.
                verbose : int, default = 0
                    controls amount of information printed to console during fit.
                n_jobs : int, default = 1
                    number of works to deploy upon execution, if applicable.
                scoring : string (sklearn evaluation metric), default =None
                    number of works to deploy upon execution, if applicable.
                refit : boolean, default=True
                    dictates whether method is refit with the best model upon grid seach completion.
                n_iter : int, default = 15
                    number of random permutations to evaluate.
            returns:
                rgs : randomized_GridSearchCV object
                    full randomized_GridSearchCV object.
        """
        for key in self.keys:
            print("running RandomizedSearchCV for {0}".format(key))
            model = self.models[key]
            params = self.params[key]
            rgs = RandomizedSearchCV(
                model,
                params,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                scoring=scoring,
                refit=refit,
                return_train_score=True,
                n_iter=n_iter,
            )
            rgs.fit(x, y)
            self.grid_searches[key] = rgs
        return rgs

    def score_summary(self, sort_by="mean_score"):
        """
        documentation:
            definition:
                method for performing randomized_GridSearchCV on multiple models, each with its own
                parameter grid, in a single execution.
            parameters:
                sort_by : string (columns of score summary)
                    column on which to sort score summary.
            returns:
                df : pandas DataFrame
                    DataFrame containing results summary of gridsearch_cv or randomized_GridSearchCV.
        """

        def row(key, scores, params):
            d = {
                "estimator": key,
                "min_score": min(scores),
                "max_score": max(scores),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
            }
            return pd.series({**params, **d})

        rows = []
        for k in self.grid_searches:
            # print(k)
            params = self.grid_searches[k].cv_results_["params"]
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).t.sort_values([sort_by], ascending=False)

        columns = ["estimator", "min_score", "mean_score", "max_score", "std_score"]
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


def PowerGridModelBuilder(self, results, model_ix):
    """
    documentation:
        description:
            model building utility to be used with power_grid_search summary results. returns
            estimator and dictionary of 'parameter : value' pairs to be passed as kwarg
            to model. used to efficiently reinstantiated specific models.
        parameters:
            results : pandas DataFrame
                DataFrame containing score summary generated by PowerGridSearcher.
            model_ix : int
                row index of specific model summarized in results DataFrame.
        returns:
            estimator : sklearn model
                sklearn model object.
            params.to_dict() : dictionary
                dictionary containing model parameters value.
    """
    estimator = results.loc[model_ix][0]
    params = results.loc[model_ix][5:].dropna(axis=0)

    # convert floats that are effectively ints to ints
    for ix in params.index:
        if not isinstance(params[ix], str):
            if int(params[ix]) == params[ix] and isinstance(params[ix], float):
                params[ix] = params[ix].astype(np.int64)

    return estimator, params.to_dict()
