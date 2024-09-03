import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from typing import List, Optional

class LabelEncoderTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.label_encoders = {}
        object_cols = X.select_dtypes(include=['object']).columns
        for col in object_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        return self
    
    def transform(self, X):
        for col, le in self.label_encoders.items():
            X[col] = le.transform(X[col].astype(str))
        return X

class KNNImputerTransform(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y=None):
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer.fit(X)
        return self
    
    def transform(self, X):
        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        return X

class SimpleImputerTransform(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
    
    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        self.imputer.fit(X)
        return self
    
    def transform(self, X):
        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        return X

def preprocess_data_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame, steps: Optional[List[str]] = None, n_neighbors: int = 5):
    # Separating columns by type
    numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    # Pipeline steps for numeric and categorical data
    numeric_pipeline_steps = []
    categorical_pipeline_steps = []
    
    if 'simple_impute' in steps:
        numeric_pipeline_steps.append(('simple_impute_numeric', SimpleImputerTransform(strategy='mean')))
        categorical_pipeline_steps.append(('simple_impute_categorical', SimpleImputerTransform(strategy='most_frequent')))
    
    if 'label_encode' in steps:
        categorical_pipeline_steps.append(('label_encode', LabelEncoderTransform()))
    
    if 'knn_impute' in steps:
        numeric_pipeline_steps.append(('knn_impute', KNNImputerTransform(n_neighbors=n_neighbors)))

    # Creating separate pipelines for numeric and categorical features
    numeric_pipeline = Pipeline(numeric_pipeline_steps)
    categorical_pipeline = Pipeline(categorical_pipeline_steps)
    
    # Combine the pipelines
    full_pipeline = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_cols),
        ('categorical', categorical_pipeline, categorical_cols)
    ])
    
    # Apply the full pipeline to both train and test data
    concat_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    processed_df = full_pipeline.fit_transform(concat_df)
    processed_df = pd.DataFrame(processed_df, columns=numeric_cols + categorical_cols)
    
    # Split the processed data back into train and test sets
    train_size = len(train_df)
    train_processed = processed_df.iloc[:train_size].reset_index(drop=True)
    test_processed = processed_df.iloc[train_size:].reset_index(drop=True)
    
    return train_processed, test_processed

if __name__ == "__main__":
    # Example usage
    train = pd.DataFrame(...)  # Replace with actual train dataframe
    test = pd.DataFrame(...)   # Replace with actual test dataframe
    
    # User-defined steps for preprocessing
    steps = ['simple_impute', 'label_encode', 'knn_impute']  # Example list, modify as needed
    n_neighbors = 5  # Example value, modify as needed
    
    train_processed, test_processed = preprocess_data_pipeline(train, test, steps=steps, n_neighbors=n_neighbors)
    print(train_processed.head())
    print(test_processed.head())
