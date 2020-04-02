import joblib
import os

dir = os.path.dirname(os.path.realpath(__file__))


def attrition():
    """
    Documentation:

        ---
        Description:
            Load IBM Employee Attrition dataset.
    """
    data = joblib.load(os.path.join(dir, "datasets/attrition/data.pkl"))
    return data


def housing():
    """
    Documentation:

        ---
        Description:
            Load Kaggle Housing Prices training dataset and validation dataset.
    """
    train = joblib.load(os.path.join(dir, "datasets/housing//train.pkl"))
    test = joblib.load(os.path.join(dir, "datasets/housing/test.pkl"))
    return train, test


def titanic():
    """
    Documentation:

        ---
        Description:
            Load Kaggle Titanic Survivorship training dataset and validation dataset.
    """
    train = joblib.load(os.path.join(dir, "datasets/titanic/train.pkl"))
    test = joblib.load(os.path.join(dir, "datasets/titanic/test.pkl"))
    return train, test

