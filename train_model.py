import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("train.csv")

# Drop unwanted columns
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Split
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Columns
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
cat_cols = ['Sex', 'Embarked', 'Pclass']

# Pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Model pipeline
from sklearn.pipeline import Pipeline as FullPipeline

model_pipeline = FullPipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

# Train
model_pipeline.fit(X, y)

# Save
with open("model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("Model saved successfully!")