from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np
                

class EstimatorSelectionHelper:
    
    
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError('Some estimators are missing parameters: {0}'.format(missing_params))
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    # Full GridSearchCV
    def fitGs(self, X, y, cv = 5, n_jobs = 1, verbose = 0, scoring = None, refit = True):
        for key in self.keys:
            print('Running GridSearchCV for {0}'.format(key))
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model
                              ,params
                              ,cv = cv
                              ,n_jobs = n_jobs
                              ,verbose = verbose
                              ,scoring = scoring
                              ,refit = refit
                              ,return_train_score = True)
            gs.fit(X,y)
            self.grid_searches[key] = gs    
        return gs
    
    # RandomizedSearchCV
    def fitRgs(self, X, y, cv = 5, n_jobs = 1, verbose = 0, scoring = None, refit = True, n_iter = 15):
        for key in self.keys:
            print('Running RandomizedSearchCV for {0}'.format(key))
            model = self.models[key]
            params = self.params[key]        
            rgs = RandomizedSearchCV(model
                                    ,params
                                    ,cv = cv
                                    ,n_jobs = n_jobs
                                    ,verbose = verbose
                                    ,scoring = scoring
                                    ,refit = refit
                                    ,return_train_score = True
                                    ,n_iter = n_iter)
            rgs.fit(X,y)
            self.grid_searches[key] = rgs    
        return rgs
        
    def scoreSummary(self, sort_by = 'mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key
                 ,'min_score': min(scores)
                 ,'max_score': max(scores)
                 ,'mean_score': np.mean(scores)
                 ,'std_score': np.std(scores)
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            #print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = 'split{}_test_score'.format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis = 1).T.sort_values([sort_by], ascending = False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
