
import sklearn.model_selection as model_selection

def rmsleCV(self, estimator, X, y, scoring, cv, modelDesc):
    cv = model_selection.cross_val_score(estimator = estimator
                                   ,X = X
                                   ,y = y
                                   ,scoring = scoring
                                   ,cv = cv)
    cv = np.sqrt(cv * -1)
    print('{}:\n mean RMSLE = {}\n std RMSLE = {}'.format(modelDesc, np.round(cv.mean(),5), np.round(cv.std(),5)))
    return cv