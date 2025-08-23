"""
Tabular Classification Baseline: Random Forest on Titanic Dataset
================================================================

This script demonstrates why Random Forest is an excellent baseline for tabular data:
1. Handles mixed data types (numerical + categorical)
2. Built-in feature importance
3. Robust to outliers and missing values
4. No feature scaling required
5. Good out-of-the-box performance

Dataset: Titanic Survival Prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def load_and_explore_data():
    """Load Titanic dataset and perform initial exploration"""
    print("=" * 60)
    print("STEP 1: DATA LOADING AND EXPLORATION")
    print("=" * 60)

    # Load the famous Titanic dataset
    titanic = sns.load_dataset("titanic")
    print(f"Dataset shape: {titanic.shape}")
    print(f"Columns: {list(titanic.columns)}")

    print("\nFirst few rows:")
    print(titanic.head())

    print("\nTarget variable distribution:")
    survival_counts = titanic["survived"].value_counts()
    print(survival_counts)
    print(f"Survival rate: {titanic['survived'].mean():.3f}")

    print("\nMissing values:")
    missing_data = titanic.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    print(missing_data)

    print("\nData types:")
    print(titanic.dtypes)

    return titanic


def preprocess_data(df):
    """Simple preprocessing - why Random Forest handles this well"""
    print("\n" + "=" * 60)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 60)

    print("Random Forest advantages for preprocessing:")
    print("✓ Handles missing values internally")
    print("✓ No need for feature scaling")
    print("✓ Can work with categorical variables (with encoding)")

    # Create a copy for processing
    data = df.copy()
    print(data.head())

    # Simple feature engineering
    print("\nCreating new features...")
    data["family_size"] = data["sibsp"] + data["parch"] + 1
    data["is_alone"] = (data["family_size"] == 1).astype(int)

    # Select features for modeling
    features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
        "family_size",
        "is_alone",
    ]

    # Handle missing values (Random Forest can handle some, but let's be explicit)
    print(f"\nFilling missing values...")
    data["age"].fillna(data["age"].median(), inplace=True)
    data["embarked"].fillna(data["embarked"].mode()[0], inplace=True)
    data["fare"].fillna(data["fare"].median(), inplace=True)

    # Encode categorical variables
    print("Encoding categorical variables...")
    le_dict = {}
    categorical_features = ["sex", "embarked"]

    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        le_dict[feature] = le
        print(f"  {feature}: {list(le.classes_)}")

    X = data[features]
    y = data["survived"]

    print(f"\nFinal feature matrix shape: {X.shape}")
    print("Features:", list(X.columns))

    return X, y, le_dict


def establish_naive_baseline(y):
    """Establish naive baselines to compare against"""
    print("\n" + "=" * 60)
    print("STEP 3: NAIVE BASELINES")
    print("=" * 60)

    # Most frequent class baseline
    most_frequent_accuracy = max(y.value_counts()) / len(y)
    print(f"Most Frequent Class Baseline: {most_frequent_accuracy:.3f}")
    print("(Always predict the most common class - 'did not survive')")

    # Random baseline
    random_accuracy = 0.5  # Binary classification
    print(f"Random Baseline: {random_accuracy:.3f}")
    print("(Random guessing)")

    print(f"\nOur goal: Beat {most_frequent_accuracy:.3f} significantly!")

    return most_frequent_accuracy


def train_random_forest_baseline(X, y):
    """Train Random Forest with default parameters"""
    print("\n" + "=" * 60)
    print("STEP 4: RANDOM FOREST BASELINE")
    print("=" * 60)

    print("Why Random Forest for tabular data baseline?")
    print("✓ Robust to overfitting")
    print("✓ Handles feature interactions automatically")
    print("✓ Provides feature importance")
    print("✓ Good default hyperparameters")
    print("✓ Fast to train")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train Random Forest with default parameters
    print("\nTraining Random Forest (default parameters)...")
    rf_baseline = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_baseline.fit(X_train, y_train)

    # Evaluate with cross-validation
    print("Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(rf_baseline, X_train, y_train, cv=5, scoring="accuracy")

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Test set evaluation
    train_accuracy = rf_baseline.score(X_train, y_train)
    test_accuracy = rf_baseline.score(X_test, y_test)

    print(f"\nTraining accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    print(f"Overfitting check: {train_accuracy - test_accuracy:.3f} difference")

    # Feature importance
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": rf_baseline.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())

    # Predictions and detailed metrics
    y_pred = rf_baseline.predict(X_test)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return rf_baseline, X_train, X_test, y_train, y_test, test_accuracy


def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """Demonstrate how to improve the baseline with hyperparameter tuning"""
    print("\n" + "=" * 60)
    print("STEP 5: HYPERPARAMETER TUNING (QUICK IMPROVEMENT)")
    print("=" * 60)

    print("Let's see if we can improve our baseline with hyperparameter tuning...")

    # Define parameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Grid search with cross-validation
    print("\nPerforming Grid Search (this may take a moment)...")
    rf_grid = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf_grid, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

    # Evaluate tuned model
    tuned_train_accuracy = grid_search.score(X_train, y_train)
    tuned_test_accuracy = grid_search.score(X_test, y_test)

    print(f"Tuned model - Training accuracy: {tuned_train_accuracy:.3f}")
    print(f"Tuned model - Test accuracy: {tuned_test_accuracy:.3f}")

    return grid_search.best_estimator_, tuned_test_accuracy


def compare_results(baseline_accuracy, tuned_accuracy, naive_accuracy):
    """Compare all our results"""
    print("\n" + "=" * 60)
    print("STEP 6: RESULTS COMPARISON")
    print("=" * 60)

    print("Performance Summary:")
    print(f"Naive Baseline (most frequent): {naive_accuracy:.3f}")
    print(f"Random Forest Baseline:        {baseline_accuracy:.3f}")
    print(f"Tuned Random Forest:           {tuned_accuracy:.3f}")

    improvement_from_naive = baseline_accuracy - naive_accuracy
    improvement_from_tuning = tuned_accuracy - baseline_accuracy

    print(f"\nImprovements:")
    print(
        f"RF Baseline vs Naive: +{improvement_from_naive:.3f} ({improvement_from_naive/naive_accuracy*100:.1f}%)"
    )
    print(
        f"Tuning vs Baseline:   +{improvement_from_tuning:.3f} ({improvement_from_tuning/baseline_accuracy*100:.1f}%)"
    )

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("✓ Random Forest provides excellent baseline performance")
    print("✓ Minimal preprocessing required")
    print("✓ Built-in feature importance for interpretability")
    print("✓ Hyperparameter tuning can provide additional gains")
    print("✓ Good balance of simplicity and performance")


def main():
    """Main execution function"""
    print("RANDOM FOREST BASELINE FOR TABULAR CLASSIFICATION")
    print("Titanic Dataset: Predicting Passenger Survival")

    # Step 1: Load and explore data
    titanic_df = load_and_explore_data()

    # Step 2: Preprocess data
    X, y, encoders = preprocess_data(titanic_df)

    # Step 3: Establish naive baseline
    naive_accuracy = establish_naive_baseline(y)

    # Step 4: Train Random Forest baseline
    rf_model, X_train, X_test, y_train, y_test, baseline_accuracy = (
        train_random_forest_baseline(X, y)
    )

    # Step 5: Hyperparameter tuning
    tuned_model, tuned_accuracy = hyperparameter_tuning(
        X_train, y_train, X_test, y_test
    )

    # Step 6: Compare results
    compare_results(baseline_accuracy, tuned_accuracy, naive_accuracy)


if __name__ == "__main__":
    main()
