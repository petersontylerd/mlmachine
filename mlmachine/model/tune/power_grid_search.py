from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np


class PowerGridSearcher:
    """
    Documentation:

        ---
        Definition:
            Class capable of performing GridSearchCV and RandomizedSearchCV on
            multiple models, each with its own parameter grid, in a single execution.
            Also returns a score summary sheet.

        ---
        Parameters:
            models : dict
                Dictionary of instantiated sklearn-style models.
            params : dict of dicts
                Dictionary containing nested dictionaries of parameter grids for each model.
    """

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "some estimators are missing Parameters: {0}".format(missing_params)
            )
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    # full GridSearchCV
    def fit_multi_gs_cv(self, x, y, cv=5, n_jobs=1, verbose=0, scoring=None, refit=True):
        """
        Documentation:

            ---
            Definition:
                Method for performing GridSearchCV on multiple models, each with its own parameter
                grid, in a single execution.

            ---
            Parameters:
                x : array
                    Training data observation.
                y : array
                    Training target data.
                cv : int, default=5
                    The number of folds to use in cross-validation procedure.
                n_jobs : int, default=1
                    Number of workers to deploy upon execution, if applicable.
                verbose : int, default=0
                    Controls amount of information printed to console during fit.
                scoring : str (sklearn evaluation metric), default=None
                    Scoring metric to use in evaluation.
                refit : bool, default=True
                    Dictates whether method is refit with the best model upon grid seach completion.

            ---
            Returns:
                gs : GridSearchCV object
                    Full GridSearchCV object.
        """
        # iterate through all estimators
        for key in self.keys:
            print("running GridSearchCV for {0}".format(key))

            # separate current model and associated param grid diciontary
            model = self.models[key]
            params = self.params[key]

            # instantiate GridSearchCV object and fit
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

            # store GridSearchCV object
            self.grid_searches[key] = gs
        return gs

    # RandomizedSearchCV
    def fit_multi_rgs_cv(self, x, y, cv=5, n_jobs=1, verbose=0, scoring=None, refit=True, n_iter=15):
        """
        Documentation:

            ---
            Definition:
                method for performing randomized_GridSearchCV on multiple models, each with its own parameter
                grid, in a single execution.

            ---
            Parameters:
                x : array
                    Training data observation.
                y : array
                    Training target data.
                cv : int, default=5
                    The number of folds to use in cross-validation procedure.
                n_jobs : int, default=1
                    Number of workers to deploy upon execution, if applicable.
                verbose : int, default=0
                    Controls amount of information printed to console during fit.
                scoring : str (sklearn evaluation metric), default=None
                    Scoring metric to use in evaluation.
                refit : bool, default=True
                    Dictates whether method is refit with the best model upon grid seach completion.
                n_iter : int, default=15
                    Number of random permutations to evaluate.

            ---
            Returns:
                rgs : randomized_GridSearchCV object
                    Full randomized_GridSearchCV object.
        """
        # iterate through all estimators
        for key in self.keys:
            print("running RandomizedSearchCV for {0}".format(key))

            # separate current model and associated param grid diciontary
            model = self.models[key]
            params = self.params[key]

            # instantiate GridSearchCV object and fit
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

            # store GridSearchCV object
            self.grid_searches[key] = rgs
        return rgs

    def score_summary(self, sort_by="mean_score"):
        """
        Documentation:

            ---
            Definition:
                Method for performing RandomizedGridSearchCV on multiple models, each with
                its own parameter grid, in a single execution.

            ---
            Parameters:
                sort_by : string (columns of score summary), default="mean_score"
                    Column on which to sort score summary.

            ---
            Returns:
                df : Pandas DataFrame
                    DataFrame containing results summary of GridSearchCV or
                    RandomizedGridSearchCV.
        """

        # helper function for collecting single row of summary data
        def row(key, scores, params):
            d = {
                "estimator": key,
                "min_score": min(scores),
                "max_score": max(scores),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []

        # iterate through grid search object
        for k in self.grid_searches:
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
    Documentation:
        
        ---
        Description:
            Model building utility to be used with power_grid_search summary results.
            Returns estimator and dictionary of 'parameter : value' pairs to be passed
            as kwarg to model. Used to efficiently reinstantiated specific models.
        
        ---
        Parameters:
            results : Pandas DataFrame
                DataFrame containing score summary generated by PowerGridSearcher.
            model_ix : int
                Row index of specific model summarized in results DataFrame.
        
        ---
        Returns:
            estimator : sklearn model
                sklearn model object.
            params.to_dict() : dictionary
                Dictionary containing model parameters value.
    """
    estimator = results.loc[model_ix][0]
    params = results.loc[model_ix][5:].dropna(axis=0)

    # convert floats that are effectively ints to ints
    for ix in params.index:
        if not isinstance(params[ix], str):
            if int(params[ix]) == params[ix] and isinstance(params[ix], float):
                params[ix] = params[ix].astype(np.int64)

    return estimator, params.to_dict()
