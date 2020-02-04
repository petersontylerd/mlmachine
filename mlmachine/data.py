import joblib
import os

dir = os.path.dirname(os.path.realpath(__file__))


def attrition():
    data = joblib.load(os.path.join(dir, "datasets\\attrition\\data.pkl"))
    return data


def housing():
    train = joblib.load(os.path.join(dir, "datasets\\housing\\train.pkl"))
    test = joblib.load(os.path.join(dir, "datasets\\housing\\test.pkl"))
    return train, test


def titanic():
    train = joblib.load(os.path.join(dir, "datasets\\titanic\\train.pkl"))
    test = joblib.load(os.path.join(dir, "datasets\\titanic\\test.pkl"))
    return train, test

