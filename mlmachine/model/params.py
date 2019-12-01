# model/parameter space
all_space = {
    "lightgbm.lgbm_classifier": {
        "class_weight": hp.choice("class_weight", [none, "balanced"]),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "boosting_type": hp.choice("boosting_type", ["gbdt", "dart", "goss"])
        # ,'boosting_type': hp.choice('boosting_type'
        #                    ,[{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}
        #                    ,{'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)}
        #                    ,{'boosting_type': 'goss', 'subsample': 1.0}])
        ,
        "learning_rate": hp.uniform("learning_rate", 0.000001, 0.2),
        "max_depth": hp.choice("max_depth", np.arange(2, 20, dtype=int)),
        "min_child_samples": hp.uniform("min_child_samples", 20, 500),
        "n_estimators": hp.choice("n_estimators", np.arange(100, 10000, 10, dtype=int)),
        "num_leaves": hp.uniform("num_leaves", 8, 150),
        "reg_alpha": hp.uniform("reg_alpha", 0.0, 1.0),
        "reg_lambda": hp.uniform("reg_lambda", 0.0, 1.0),
        "subsample_for_bin": hp.uniform("subsample_for_bin", 20000, 400000),
    },
    "linear_model.logistic_regression": {
        "c": hp.loguniform("c", np.log(0.001), np.log(0.2)),
        "penalty": hp.choice("penalty", ["l1", "l2"]),
    },
    "xgboost.xgb_classifier": {
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "gamma": hp.uniform("gamma", 0.0, 10),
        "learning_rate": hp.uniform("learning_rate", 0.000001, 0.2),
        "max_depth": hp.choice("max_depth", np.arange(2, 20, dtype=int)),
        "min_child_weight": hp.uniform("min_child_weight", 1, 20),
        "n_estimators": hp.choice("n_estimators", np.arange(100, 10000, 10, dtype=int)),
        "subsample": hp.uniform("subsample", 0.5, 1),
    },
    "ensemble.random_forest_classifier": {
        "bootstrap": hp.choice("bootstrap", [True, False]),
        "max_depth": hp.choice("max_depth", np.arange(2, 20, dtype=int)),
        "n_estimators": hp.choice("n_estimators", np.arange(100, 10000, 10, dtype=int)),
        "max_features": hp.choice("max_features", ["auto", "sqrt"]),
        "min_samples_split": hp.choice(
            "min_samples_split", np.arange(2, 40, dtype=int)
        ),
        "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(2, 40, dtype=int)),
    },
    "ensemble.gradient_boosting_classifier": {
        "n_estimators": hp.choice("n_estimators", np.arange(100, 10000, 10, dtype=int)),
        "max_depth": hp.choice("max_depth", np.arange(2, 20, dtype=int)),
        "max_features": hp.choice("max_features", ["auto", "sqrt"]),
        "learning_rate": hp.uniform("learning_rate", 0.000001, 0.2),
        "loss": hp.choice("loss", ["deviance", "exponential"]),
        "min_samples_split": hp.choice(
            "min_samples_split", np.arange(2, 40, dtype=int)
        ),
        "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(2, 40, dtype=int)),
    },
    "ensemble.ada_boost_classifier": {
        "n_estimators": hp.choice("n_estimators", np.arange(100, 10000, 10, dtype=int)),
        "learning_rate": hp.uniform("learning_rate", 0.000001, 0.2),
        "algorithm": hp.choice("algorithm", ["samme", "samme.r"]),
    },
    "ensemble.extra_trees_classifier": {
        "n_estimators": hp.choice("n_estimators", np.arange(100, 10000, 10, dtype=int)),
        "max_depth": hp.choice("max_depth", np.arange(2, 20, dtype=int)),
        "min_samples_split": hp.choice(
            "min_samples_split", np.arange(2, 40, dtype=int)
        ),
        "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(2, 40, dtype=int)),
        "max_features": hp.choice("max_features", ["auto", "sqrt"]),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
    },
    "svm.svc": {
        "c": hp.uniform("c", 0.00001, 10),
        "decision_function_shape": hp.choice("decision_function_shape", ["ovo", "ovr"]),
        "gamma": hp.uniform("gamma", 0.00001, 10),
    },
    "neighbors.k_neighbors_classifier": {
        "algorithm": hp.choice("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "n_neighbors": hp.choice("n_neighbors", np.arange(1, 20, dtype=int)),
        "weights": hp.choice("weights", ["distance", "uniform"]),
    },
}


# model/parameter space
all_space = {
    "linear_model.lasso": {"alpha": hp.uniform("alpha", 0.0000001, 20)},
    "linear_model.ridge": {"alpha": hp.uniform("alpha", 0.0000001, 20)},
    "linear_model.elastic_net": {
        "alpha": hp.uniform("alpha", 0.0000001, 20),
        "l1_ratio": hp.uniform("l1_ratio", 0.0, 0.2),
    },
    "kernel_ridge.kernel_ridge": {
        "alpha": hp.uniform("alpha", 0.000001, 15),
        "kernel": hp.choice("kernel", ["linear", "polynomial", "rbf"]),
        "degree": hp.choice("degree", [2, 3]),
        "gamma": hp.uniform("gamma", 0.0, 10),
    },
    "lightgbm.lgbm_regressor": {
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "boosting_type": hp.choice("boosting_type", ["gbdt", "dart", "goss"])
        # ,'boosting_type': hp.choice('boosting_type'
        #                    ,[{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}
        #                    ,{'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)}
        #                    ,{'boosting_type': 'goss', 'subsample': 1.0}])
        ,
        "learning_rate": hp.uniform("learning_rate", 0.000001, 0.2),
        "max_depth": hp.choice("max_depth", np.arange(2, 20, dtype=int)),
        "min_child_samples": hp.uniform("min_child_samples", 20, 500),
        "n_estimators": hp.choice("n_estimators", np.arange(100, 10000, 10, dtype=int)),
        "num_leaves": hp.uniform("num_leaves", 8, 150),
        "reg_alpha": hp.uniform("reg_alpha", 0.0, 1.0),
        "reg_lambda": hp.uniform("reg_lambda", 0.0, 1.0),
        "subsample_for_bin": hp.uniform("subsample_for_bin", 20000, 400000),
    },
    "xgboost.xgb_regressor": {
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "gamma": hp.uniform("gamma", 0.0, 10),
        "reg_alpha": hp.uniform("reg_alpha", 0.0, 1.0),
        "reg_lambda": hp.uniform("reg_lambda", 0.0, 1.0),
        "learning_rate": hp.uniform("learning_rate", 0.000001, 0.2),
        "max_depth": hp.choice("max_depth", np.arange(2, 20, dtype=int)),
        "min_child_weight": hp.uniform("min_child_weight", 1, 20),
        "n_estimators": hp.choice("n_estimators", np.arange(100, 10000, 10, dtype=int)),
        "subsample": hp.uniform("subsample", 0.5, 1),
    },
    "ensemble.random_forest_regressor": {
        "bootstrap": hp.choice("bootstrap", [True, False]),
        "max_depth": hp.choice("max_depth", np.arange(2, 20, dtype=int)),
        "n_estimators": hp.choice("n_estimators", np.arange(100, 10000, 10, dtype=int)),
        "max_features": hp.choice("max_features", ["auto", "sqrt"]),
        "min_samples_split": hp.choice(
            "min_samples_split", np.arange(2, 40, dtype=int)
        ),
        "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(2, 40, dtype=int)),
    },
    "ensemble.gradient_boosting_regressor": {
        "n_estimators": hp.choice("n_estimators", np.arange(100, 10000, 10, dtype=int)),
        "max_depth": hp.choice("max_depth", np.arange(2, 20, dtype=int)),
        "max_features": hp.choice("max_features", ["auto", "sqrt"]),
        "learning_rate": hp.uniform("learning_rate", 0.000001, 0.2),
        "loss": hp.choice("loss", ["ls", "lad", "huber", "quantile"]),
        "min_samples_split": hp.choice(
            "min_samples_split", np.arange(2, 40, dtype=int)
        ),
        "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(2, 40, dtype=int)),
    },
    "ensemble.ada_boost_regressor": {
        "n_estimators": hp.choice("n_estimators", np.arange(100, 10000, 10, dtype=int)),
        "learning_rate": hp.uniform("learning_rate", 0.000001, 0.2),
        "loss": hp.choice("loss", ["linear", "square", "exponential"]),
    },
    "ensemble.extra_trees_regressor": {
        "n_estimators": hp.choice("n_estimators", np.arange(100, 10000, 10, dtype=int)),
        "max_depth": hp.choice("max_depth", np.arange(2, 20, dtype=int)),
        "min_samples_split": hp.choice(
            "min_samples_split", np.arange(2, 40, dtype=int)
        ),
        "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(2, 40, dtype=int)),
        "max_features": hp.choice("max_features", ["auto", "sqrt"]),
    },
    "svm.svr": {
        "c": hp.uniform("c", 0.00001, 10),
        "kernel": hp.choice("kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "degree": hp.choice("degree", [2, 3]),
        "gamma": hp.uniform("gamma", 0.0001, 10),
        "epsilon": hp.uniform("epsilon", 0.001, 5),
    },
    "neighbors.k_neighbors_regressor": {
        "algorithm": hp.choice("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "n_neighbors": hp.choice("n_neighbors", np.arange(1, 20, dtype=int)),
        "weights": hp.choice("weights", ["distance", "uniform"]),
        "p": hp.choice("p", [1, 2]),
    },
}
