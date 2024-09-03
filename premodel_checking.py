import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, 
    RandomForestRegressor, GradientBoostingRegressor, 
    BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, RidgeClassifierCV, 
    SGDClassifier, Ridge, SGDRegressor, ElasticNet
)
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier as XgbClassifier, XGBRegressor as XgbRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold

# Function to evaluate classification models
def classify_models(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    
    model_name = model.__class__.__name__
    return model_name, round(np.mean(accuracy_scores), 5)

# Function to evaluate regression models
def regress_models(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse_scores.append(mean_squared_error(y_test, y_pred))
    
    model_name = model.__class__.__name__
    return model_name, round(np.mean(mse_scores), 5)

# Main function that can be run as standalone or imported as a module
def main(X, y):
    # Ask the user to specify the problem type
    problem_type = input("Is this a classification or regression problem? (Enter 'classification' or 'regression'): ").strip().lower()

    if problem_type == 'classification':
        classifiers = [
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            BaggingClassifier(),
            AdaBoostClassifier(),
            ExtraTreesClassifier(),
            MLPClassifier(),
            KNeighborsClassifier(),
            LogisticRegression(),
            RidgeClassifier(),
            RidgeClassifierCV(),
            SGDClassifier(),
            LGBMClassifier(),
            XgbClassifier(),
            CatBoostClassifier(silent=True)
        ]

        scores = []
        for clf in classifiers:
            scores.append(classify_models(clf, X, y))

        results_df = pd.DataFrame(scores, columns=['Model', 'Accuracy'])
    
    elif problem_type == 'regression':
        regressors = [
            DecisionTreeRegressor(),
            RandomForestRegressor(),
            GradientBoostingRegressor(),
            BaggingRegressor(),
            AdaBoostRegressor(),
            ExtraTreesRegressor(),
            MLPRegressor(),
            KNeighborsRegressor(),
            Ridge(),
            SGDRegressor(),
            ElasticNet(),
            LGBMRegressor(),
            XgbRegressor(),
            CatBoostRegressor(silent=True)
        ]

        scores = []
        for reg in regressors:
            scores.append(regress_models(reg, X, y))

        results_df = pd.DataFrame(scores, columns=['Model', 'Mean Squared Error'])
    
    else:
        raise ValueError("Invalid input. Please enter 'classification' or 'regression'.")

    # Display results
    print(results_df)

# If the script is run directly, you can replace the below code with your actual dataset
if __name__ == "__main__":
    # For module usage, the user will provide X and y as inputs
    X = pd.DataFrame(...)  # Replace with actual feature dataset
    y = pd.Series(...)     # Replace with actual target dataset
    main(X, y)
