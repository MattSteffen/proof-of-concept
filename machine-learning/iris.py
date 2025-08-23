import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for easier data manipulation
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Display basic information about the dataset
print(df.info())
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data Cleaning
# Note: The Iris dataset is typically clean, but we'll include steps for completeness

# 1. Handle missing values (if any)
# We'll use SimpleImputer in our pipeline, so we don't need to do this manually
# But if we wanted to, we could do:
# df = df.fillna(df.mean())

# 2. Check for outliers
# We'll use a simple IQR method to detect outliers
def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers

# Detect outliers
outliers_to_drop = detect_outliers(df, 2, iris.feature_names)
print(f"\nNumber of outliers detected: {len(outliers_to_drop)}")

# We won't remove outliers in this case, but if we wanted to, we could do:
# df = df.drop(outliers_to_drop, axis=0).reset_index(drop=True)

# Prepare data for ML

# Define features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessor and classifier
clf = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Feature importance
feature_importance = clf.named_steps['classifier'].feature_importances_
feature_names = iris.feature_names

importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Predict class for a specific flower
# Example: Sepal length: 5.1, Sepal width: 3.5, Petal length: 1.4, Petal width: 0.2
sample_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction = clf.predict(sample_flower)
prediction_proba = clf.predict_proba(sample_flower)

print(f"\nPredicted class for the sample flower: {iris.target_names[prediction[0]]}")
print("Class probabilities:")
for i, prob in enumerate(prediction_proba[0]):
    print(f"{iris.target_names[i]}: {prob:.2f}")
