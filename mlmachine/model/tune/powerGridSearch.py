from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np


class PowerGridSearcher:
    """
    Documentation:
        Definition:
            Class capable of performing GridSearchCV and RandomizedSearchCV on
            multiple models, each with its own parameter grid, in a single execution.
            Also returns a score summary sheet.
        Parameters:
            models : dict
                Dictionary of instantiated sklearn models.
            params : dict of dicts
                Dictionary containing nested dictionaries of parameter grids for each model.
    """

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "Some estimators are missing parameters: {0}".format(missing_params)
            )
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    # Full GridSearchCV
    def fitMultiGsCV(self, X, y, cv=5, n_jobs=1, verbose=0, scoring=None, refit=True):
        """
        Documentation:
            Definition:
                Method for performing GridSearchCV on multiple models, each with its own parameter 
                grid, in a single execution. 
            Parameters:
                X : array
                    Input dataset.
                y : array
                    Input dataset labels.
                cv : int, default = 5
                    The of folds to perform in cross-validation.
                verbose : int, default = 0
                    Controls amount of information printed to console during fit.
                n_jobs : int, default = 1
                    Number of works to deploy upon execution, if applicable.
                scoring : string (sklearn evaluation metric), default = None
                    Number of works to deploy upon execution, if applicable.
                refit : boolean, default = True
                    Dictates whether method is refit with the best model upon grid seach completion.
            Returns:
                gs : GridSearchCV object
                    Full GridSearchCV object.
        """        
        for key in self.keys:
            print("Running GridSearchCV for {0}".format(key))
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
            gs.fit(X, y)
            self.grid_searches[key] = gs
        return gs

    # RandomizedSearchCV
    def fitMultiRgsCV(
        self, X, y, cv=5, n_jobs=1, verbose=0, scoring=None, refit=True, n_iter=15
    ):
        """
        Documentation:
            Definition:
                Method for performing RandomizedGridSearchCV on multiple models, each with its own parameter 
                grid, in a single execution. 
            Parameters:
                X : array
                    Input dataset.
                y : array
                    Input dataset labels.
                cv : int, default = 5
                    The of folds to perform in cross-validation.
                verbose : int, default = 0
                    Controls amount of information printed to console during fit.
                n_jobs : int, default = 1
                    Number of works to deploy upon execution, if applicable.
                scoring : string (sklearn evaluation metric), default = None
                    Number of works to deploy upon execution, if applicable.
                refit : boolean, default = True
                    Dictates whether method is refit with the best model upon grid seach completion.
                n_iter : int, default = 15
                    Number of random permutations to evaluate.
            Returns:
                rgs : RandomizedGridSearchCV object
                    Full RandomizedGridSearchCV object.
        """
        for key in self.keys:
            print("Running RandomizedSearchCV for {0}".format(key))
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
            rgs.fit(X, y)
            self.grid_searches[key] = rgs
        return rgs

    def scoreSummary(self, sort_by="mean_score"):
        """
        Documentation:
            Definition:
                Method for performing RandomizedGridSearchCV on multiple models, each with its own 
                parameter grid, in a single execution.
            Parameters:
                sort_by : string (columns of score summary)
                    Column on which to sort score summary.
            Returns:
                df : Pandas DataFrame
                    DataFrame containing results summary of GridsearchCV or RandomizedGridSearchCV.
        """        
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

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ["estimator", "min_score", "mean_score", "max_score", "std_score"]
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


def powerGridModelBuilder(self, results, modelIx):
    """
    Documentation:
        Description:
            Model building utility to be used with powerGridSearch summary results. Returns 
            estimator and dictionary of 'parameter : value' pairs to be passed as kwarg
            to model. Used to efficiently reinstantiated specific models.
        Parameters:
            results : Pandas DataFrame
                DataFrame containing score summary generated by PowerGridSearcher.
            modelIx : int
                Row index of specific model summarized in results DataFrame.
        Returns:
            estimator : sklearn model
                sklearn model object.
            params.to_dict() : dictionary
                Dictionary containing model parameters value.
    """
    estimator = results.loc[modelIx][0]
    params = results.loc[modelIx][5:].dropna(axis=0)

    # convert floats that are effectively ints to ints
    for ix in params.index:
        if not isinstance(params[ix], str):
            if int(params[ix]) == params[ix] and isinstance(params[ix], float):
                params[ix] = params[ix].astype(np.int64)

    return estimator, params.to_dict()
