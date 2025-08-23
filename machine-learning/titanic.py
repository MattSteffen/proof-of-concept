import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# URL of the Titanic dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"

# Fetch and load the data
df = pd.read_csv(url)

# Display basic information about the dataset
print(df.info())
print(df.describe())

# Data Cleaning

# 1. Handle missing values
# For 'Age', we'll impute with the median
# For 'Embarked', we'll fill with the most frequent value
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 2. Drop less useful columns
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 3. Convert categorical variables to numeric
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

# 4. Create feature for family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 5. Create age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 100], labels=[0, 1, 2, 3])

# Prepare data for ML

# Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing steps
numeric_features = ['Age', 'Fare', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'AgeGroup']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessor and classifier
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = clf.named_steps['classifier'].feature_importances_
feature_names = (numeric_features + 
                 clf.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .named_steps['onehot']
                    .get_feature_names(categorical_features).tolist())

importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Predict survival probability for a specific passenger
# Example: 30-year-old female, 1st class, embarked from C, fare 100, no siblings/spouses, no parents/children
sample_passenger = pd.DataFrame({
    'Pclass': [1],
    'Sex': [0],  # female
    'Age': [30],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [100],
    'Embarked': ['C'],
    'FamilySize': [1],
    'AgeGroup': [1]  # Age group 1 (18-35)
})

survival_prob = clf.predict_proba(sample_passenger)[0][1]
print(f"\nSurvival probability for the sample passenger: {survival_prob:.2f}")
