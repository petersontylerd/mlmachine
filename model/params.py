 # parameter space
allSpace = {
            'lightgbm.LGBMClassifier' : {
                'class_weight' : hp.choice('class_weight', [None, 'balanced'])
                ,'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1.0, 0.05)
                ,'boosting_type' : hp.choice('boosting_type', ['gbdt', 'dart', 'goss'])                
                #,'boosting_type': hp.choice('boosting_type'
                #                    ,[{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}
                #                    ,{'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)}
                #                    ,{'boosting_type': 'goss', 'subsample': 1.0}])
                ,'learning_rate' : hp.quniform('learning_rate', 0.01, 0.2, 0.01)
                ,'max_depth' : hp.choice('max_depth', np.arange(2, 20, dtype = int))
                ,'min_child_samples' : hp.quniform('min_child_samples', 20, 500, 5)
                ,'n_estimators' : hp.choice('n_estimators', np.arange(100, 10000, 10, dtype = int))
                ,'num_leaves': hp.quniform('num_leaves', 8, 150, 1)
                ,'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0)
                ,'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
                ,'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000)                    
            }
            ,'linear_model.LogisticRegression' : {
                'C': hp.loguniform('C', np.log(0.001), np.log(0.2))
                ,'penalty': hp.choice('penalty', ['l1', 'l2'])
            }
            ,'xgboost.XGBClassifier' : {
                'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1.0, 0.05)
                ,'gamma' : hp.quniform('gamma', 0.0, 10, 0.05)
                ,'learning_rate' : hp.quniform('learning_rate', 0.01, 0.2, 0.01)
                ,'max_depth' : hp.choice('max_depth', np.arange(2, 20, dtype = int))
                ,'min_child_weight': hp.quniform ('min_child_weight', 1, 20, 1)
                ,'n_estimators' : hp.choice('n_estimators', np.arange(100, 10000, 10, dtype = int))
                ,'subsample': hp.uniform ('subsample', 0.5, 1)
            }
            ,'ensemble.RandomForestClassifier' : {
                'bootstrap' : hp.choice('bootstrap', [True, False])
                ,'max_depth' : hp.choice('max_depth', np.arange(2, 20, dtype = int))
                ,'n_estimators' : hp.choice('n_estimators', np.arange(100, 10000, 10, dtype = int))
                ,'max_features' : hp.choice('max_features', ['auto','sqrt'])
                ,'min_samples_split' : hp.choice('min_samples_split', np.arange(2, 40, dtype = int))
                ,'min_samples_leaf' : hp.choice('min_samples_leaf', np.arange(2, 40, dtype = int))
            }
            ,'ensemble.GradientBoostingClassifier' : {
                'n_estimators' : hp.choice('n_estimators', np.arange(100, 10000, 10, dtype = int))
                ,'max_depth' : hp.choice('max_depth', np.arange(2, 20, dtype = int))
                ,'max_features' : hp.choice('max_features', ['auto','sqrt'])    
                ,'learning_rate' : hp.quniform('learning_rate', 0.01, 0.2, 0.01)
                ,'loss' : hp.choice('loss', ['deviance','exponential'])    
                ,'min_samples_split' : hp.choice('min_samples_split', np.arange(2, 40, dtype = int))
                ,'min_samples_leaf' : hp.choice('min_samples_leaf', np.arange(2, 40, dtype = int))
            }
            ,'ensemble.AdaBoostClassifier' : {
                'n_estimators' : hp.choice('n_estimators', np.arange(100, 10000, 10, dtype = int))
                ,'learning_rate' : hp.quniform('learning_rate', 0.01, 0.2, 0.01)
                ,'algorithm' : hp.choice('algorithm', ['SAMME', 'SAMME.R'])                    
            }
            ,'naive_bayes.BernoulliNB' : {
                'alpha' :  hp.uniform('alpha', 0.01, 10)
            }
            ,'ensemble.BaggingClassifier' : {
                'n_estimators' : hp.choice('n_estimators', np.arange(100, 10000, 10, dtype = int))
                ,'max_samples' : hp.uniform('max_samples', 0.2, 1)                    
            }
            ,'ensemble.ExtraTreesClassifier' : {
                'n_estimators' : hp.choice('n_estimators', np.arange(100, 10000, 10, dtype = int))
                ,'max_depth' : hp.choice('max_depth', np.arange(2, 20, dtype = int))
                ,'min_samples_split' : hp.choice('min_samples_split', np.arange(2, 40, dtype = int))
                ,'min_samples_leaf' : hp.choice('min_samples_leaf', np.arange(2, 40, dtype = int))
                ,'max_features' : hp.choice('max_features', ['auto','sqrt'])
                ,'criterion' : hp.choice('criterion', ['gini','entropy'])
            }
            ,'svm.SVC' : {
                'C' : hp.uniform('C', 0.00001, 10)
                ,'decision_function_shape' : hp.choice('decision_function_shape', ['ovo','ovr'])
                ,'gamma' : hp.uniform('gamma', 0.00001, 10)
            }
            ,'neighbors.KNeighborsClassifier' : {
                'algorithm' : hp.choice('algorithm', ['auto','ball_tree','kd_tree','brute'])
                ,'n_neighbors' : hp.choice('n_neighbors', np.arange(1, 20, dtype = int))
                ,'weights' : hp.choice('weights', ['distance','uniform'])
            }
}
