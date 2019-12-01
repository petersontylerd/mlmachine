import joblib
import os

DIR = os.path.dirname(os.path.realpath(__file__))


def attrition():
    data = joblib.load(os.path.join(DIR, "datasets/attrition/data.pkl"))
    return data


def housing():
    train = joblib.load(os.path.join(DIR, "datasets/housing/train.pkl"))
    test = joblib.load(os.path.join(DIR, "datasets/housing/test.pkl"))
    return train, test


def titanic():
    train = joblib.load(os.path.join(DIR, "datasets/titanic/train.pkl"))
    test = joblib.load(os.path.join(DIR, "datasets/titanic/test.pkl"))
    return train, test

