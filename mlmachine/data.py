import joblib
import os

DIR = os.path.dirname(os.path.realpath(__file__))


def attrition():
	dataset = joblib.load(os.path.join(DIR, 'datasets/attrition.pkl'))
	return dataset

def housing():
	dataset = joblib.load(os.path.join(DIR, 'datasets/housing.pkl'))
	return dataset

def titanic():
	dataset = joblib.load(os.path.join(DIR, 'datasets/titanic.pkl'))
	return dataset