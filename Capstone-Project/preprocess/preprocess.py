import csv
import pandas as pd


def load_training_dataset(dataset=None):
    """
        Returns the dataframe loading the csv dataset

        Params
        ======
            dataset {str} - Filepath to the training dataset

        Returns
        =======
            pandas.DataFrame
    """

    if not dataset:
        raise ValueError("'dataset' cannot be None or Empty")

    if not isinstance(dataset, str):
        raise TypeError("'{}' should be of {}".format('dataset', str))

    return pd.read_csv(dataset)

def print_classes_per_column(dataframe=None):
    """
        Prints distinct values per column in the dataframe
    """

    # print the No. of classes per each column
    columns = list(dataframe.columns.values)

    for column in columns:
        classes = dataframe[column].unique()
        print("Classes in '{}' column are {}".format(column, set(classes)))

def get_missing_values_per_column(dataframe=None):
    """
        Returns a count of rows in each column that are empty in a dataframe
    """
    return dataframe.isnull().sum()

def drop_columns_from_dataframe(dataframe=None, columns=None):
    """
        Drops a list of columns in the dataframe
    """

    if not columns:
        return dataframe
    return dataframe.drop(labels=columns, axis=1)

def replace_with_none(value=None, column=None, dataframe=None):
    """
        Replaces the value in column inside a dataframe with None
    """
    dataframe[column] = dataframe[column].map(lambda x: x if x != value else None)

def preprocess_dataset(dataset=None):

    if not dataset:
        raise ValueError("'dataset' file path cannot be None or Empty")


    # Load the Training dataset
    dframe = load_training_dataset(dataset='./data/train/train.csv')

    # Returns the missing rows per each column in the dataset
    get_missing_values_per_column(dataframe=dframe)

    # Drop Irrelevant columns from the dataframe
    get

def main():
    pass

if __name__ == "__main__":
    main()
