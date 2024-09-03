# Here is a Python module that can be used as both a standalone script and an importable module. 
# It incorporates the AutoGluon library to perform model training and prediction on tabular datasets.

import autogluon
from autogluon.tabular import TabularDataset, TabularPredictor

def autogluon_ensemble(train, test, sub, label, eval_metric='accuracy', presets='best_quality'):
    """
    Trains an ensemble of models using AutoGluon and makes predictions on the test dataset.
    
    Args:
        train (pd.DataFrame): The training dataset.
        test (pd.DataFrame): The testing dataset.
        sub (pd.DataFrame): The submission dataframe where predictions will be stored.
        label (str): The target variable name.
        eval_metric (str): Evaluation metric for AutoGluon. Defaults to 'accuracy'.
        presets (str): Presets for AutoGluon. Defaults to 'best_quality'.

    Returns:
        pd.DataFrame: The submission dataframe with predictions.
    """

    train_data = TabularDataset(train)
    test_data = TabularDataset(test)

    predictor = TabularPredictor(label=label, eval_metric=eval_metric).fit(train_data, presets=presets)

    pred = predictor.predict(test_data)
    sub[f'{label}'] = pred

    # Save submission file
    sub.to_csv("submission_autogluon.csv", index=False)
    
    return sub

def main():
    # Example usage
    import pandas as pd

    # Example data loading (replace with your actual data)
    train = pd.read_csv('path_to_train.csv')
    test = pd.read_csv('path_to_test.csv')
    sub = pd.read_csv('path_to_sample_submission.csv')

    label = 'price'  # Replace with your target column name
    eval_metric = 'mean_squared_error'  # Replace with the appropriate metric for your task
    presets = 'best_quality'  # Use 'best_quality', 'high_quality_fast_inference_only_refit', etc.

    # Call the AutoGluon function
    submission_df = autogluon_ensemble(train, test, sub, label, eval_metric, presets)

    # Print the first few rows of the submission file
    print(submission_df.head())

if __name__ == "__main__":
    main()


