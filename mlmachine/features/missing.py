import numpy as np
import pandas as pd


def missing_data_dropper_all(self, data):
    """
    Documentation:
        Description:
            drops columns that include any missing data from self.data.
        Parameters:
            data : Pandas DataFrame, default=None
                Pandas DataFrame containing independent variables.
        Returns:
            data : Pandas DataFrame
                modified input with all columns containing missing data removed.
    """
    missing_columns = data.isnull().sum()
    missing_columns = missing_columns[missing_columns > 0].index
    data = data.drop(missing_columns, axis=1)

    return data


def missing_col_compare(self, train_data, validation_data):
    """
    Documentation:
        Description:
            compares the columns that contain missing data in the training dataset
            to the columns that contain missing data in the validation dataset. prints
            a summary of the disjunction.
        Parameters:
            train_data : Pandas DataFrame
                Pandas DataFrame containing training data.
            validation_data : Pandas DataFrame
                Pandas DataFrame containing validation data.
    """
    train_missing = train_data.isnull().sum()
    train_missing = train_missing[train_missing > 0].index

    validation_missing = validation_data.isnull().sum()
    validation_missing = validation_missing[validation_missing > 0].index

    print("Feature has missing values in validation data, not training data.")
    print(set(validation_missing) - set(train_missing))
    print("")
    print("Feature has missing values in training data, not validation data.")
    print(set(train_missing) - set(validation_missing))
