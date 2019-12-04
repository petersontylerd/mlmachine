import numpy as np
import pandas as pd


def missing_data_dropper_all(self, data):
    """
    documentation:
        description:
            drops columns that include any missing data from self.data.
        parameters:
            data : pandas DataFrame, default =None
                pandas DataFrame containing independent variables.
        returns:
            data : pandas DataFrame
                modified input with all columns containing missing data removed.
    """
    missing_columns = data.isnull().sum()
    missing_columns = missing_columns[missing_columns > 0].index
    data = data.drop(missing_columns, axis=1)

    return data


def missing_col_compare(self, train, validation):
    """
    documentation:
        description:
            compares the columns that contain missing data in the training dataset
            to the columns that contain missing data in the validation dataset. prints
            a summary of the disjunction.
        parameters:
            train : pandas DataFrame
                training dataset.
            validation : pandas DataFrame
                validation dataset.
    """
    train_missing = train.isnull().sum()
    train_missing = train_missing[train_missing > 0].index

    validation_missing = validation.isnull().sum()
    validation_missing = validation_missing[validation_missing > 0].index

    print("missing in validation, not train")
    print(set(validation_missing) - set(train_missing))
    print("")
    print("missing in train, not validation")
    print(set(train_missing) - set(validation_missing))
