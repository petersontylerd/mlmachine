
import sklearn.model_selection as model_selection

def rmsleCV(self, estimator, X, y, scoring, cv, modelDesc):
    """
    Documentation:
        Description:
            Returns cross-validation object summarizing model performance using
            root mean squared log error.
        Parameters:
            estimator : sklearn model or pipeline
                Model or pipeline to be evaluated.
            X : array
                Data array containing independent variables.
            y : array
                Data array containing dependent variable.
            scoring : sklearn measurement object
                Object for capturing model performance.
            cv : sklearn cross-validation object
                Object for proscribing how to partition data in the cross-validation
                procedure.
            modelDesc : string
                Model name.
        Returns:
            cv : sklearn cross-validation object
                Object containing summary of cross-validation results.
    """
    cv = model_selection.cross_val_score(estimator = estimator
                                   ,X = X
                                   ,y = y
                                   ,scoring = scoring
                                   ,cv = cv)
    cv = np.sqrt(cv * -1)
    print('{}:\n mean RMSLE = {}\n std RMSLE = {}'.format(modelDesc, np.round(cv.mean(),5), np.round(cv.std(),5)))
    return cv