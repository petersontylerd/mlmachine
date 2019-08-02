import sklearn.ensemble as ensemble
import sklearn.naive_bayes as naive_bayes
import sklearn.svm as svm


class SklearnHelper:
    """
    Documentation:
        Description:
            Helper class for instantiating an input model type with a provided
            parameter set.
        Parameters:
            model : sklearn model
                Model to instantiate.
            seed : int
                Random number seed.
            params : dictionary
                Dictionary containing 'parameter : value' pairs.
            nJobs : int, default = 2
                Number of works to use when training the model. This parameter will be 
                ignored if the model does not have this parameter.
        Returns:
            model : model object
                Model instantiated using parameter set. Model possesses train, predict, fit
                and feature_importances methods.
    """

    def __init__(self, model, seed=0, params=None, nJobs=2):
        # ensure that models which do not have an n_jobs parameter do not have that parameter
        # added to the parameter kwargs
        if model not in [
            ensemble.GradientBoostingClassifier,
            ensemble.GradientBoostingRegressor,
            ensemble.AdaBoostClassifier,
            ensemble.AdaBoostRegressor,
            naive_bayes.BernoulliNB,
            naive_bayes.GaussianNB,
            svm.SVC,
        ]:
            params["n_jobs"] = nJobs
        self.model = model(**params)

    def train(self, XTrain, yTrain):
        self.model.fit(XTrain, yTrain)

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.fit(x, y)

    def feature_importances(self, x, y):
        return self.model.fit(x, y).feature_importances_