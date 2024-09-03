import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, train_path, test_path, submission_path):
        """
        Initializes the DataAnalyzer class with paths to the dataset files.
        
        :param train_path: Path to the training dataset CSV file.
        :param test_path: Path to the testing dataset CSV file.
        :param submission_path: Path to the sample submission CSV file.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.submission_path = submission_path
        
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)
        self.sub = pd.read_csv(self.submission_path)
        
    def print_info(self):
        """Prints basic information about the datasets."""
        print("Train Data Information:")
        print(self.train.info())
        print("\nTest Data Information:")
        print(self.test.info())
        
    def print_shapes(self):
        """Prints the shapes of the datasets."""
        print("\nShapes:")
        print(f"Train Shape: {self.train.shape}")
        print(f"Test Shape: {self.test.shape}")
        
    def print_summary_statistics(self):
        """Prints summary statistics of the datasets."""
        print("\nTrain Data Summary Statistics:")
        print(self.train.describe(include='all'))
        print("\nTest Data Summary Statistics:")
        print(self.test.describe(include='all'))
        
    def print_first_rows(self):
        """Prints the first few rows of each DataFrame."""
        print("\nFirst Few Rows of Train Data:")
        print(self.train.head())
        print("\nFirst Few Rows of Test Data:")
        print(self.test.head())
        
    def check_missing_values(self):
        """Prints the missing values in each DataFrame."""
        print("\nMissing Values in Train Data:")
        print(self.train.isnull().sum())
        print("\nMissing Values in Test Data:")
        print(self.test.isnull().sum())
        
    def print_column_names(self):
        """Prints the column names of each DataFrame."""
        print("\nColumn Names in Train Data:")
        print(self.train.columns)
        print("\nColumn Names in Test Data:")
        print(self.test.columns)
        
    def print_data_types(self):
        """Prints the data types of each column in the DataFrames."""
        print("\nData Types in Train Data:")
        print(self.train.dtypes)
        print("\nData Types in Test Data:")
        print(self.test.dtypes)

    def drop_columns(self,columns_to_drop=None):
        if columns_to_drop is None:
            columns_to_drop = []

        self.sub.drop(columns=[],inplace=True)
        self.train.drop(columns=columns_to_drop,inplace=True)
        self.test.drop(columns=columns_to_drop,inplace=True)
        print("columns_dropped_successfully")
        return self.train, self.test,self.sub


# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    train_path = "playground-series-s4e9/train.csv"
    test_path = "playground-series-s4e9/test.csv"
    submission_path = "playground-series-s4e9/sample_submission.csv"

    analyzer = DataAnalyzer(train_path, test_path, submission_path)
    analyzer.print_info()
    analyzer.print_shapes()
    analyzer.print_summary_statistics()
    analyzer.print_first_rows()
    analyzer.check_missing_values()
    analyzer.print_column_names()
    analyzer.print_data_types()
    analyzer.drop_columns()