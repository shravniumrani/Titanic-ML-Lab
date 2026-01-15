import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Fill missing values
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

train['Fare'].fillna(train['Fare'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Convert categorical to numeric
train['Sex'] = train['Sex'].map({'male':0,'female':1})
test['Sex'] = test['Sex'].map({'male':0,'female':1})

train['Embarked'].fillna('S', inplace=True)
test['Embarked'].fillna('S', inplace=True)

train['Embarked'] = train['Embarked'].map({'S':0,'C':1,'Q':2})
test['Embarked'] = test['Embarked'].map({'S':0,'C':1,'Q':2})

# Feature selection
features = ['Pclass','Sex','Age','Fare','Embarked']
X = train[features]
y = train['Survived']

# Train-validation split (for bias-variance demonstration)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Logistic Regression (high bias)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_val)
print("Logistic Regression Accuracy:", accuracy_score(y_val, log_pred))

# Model 2: Random Forest (lower bias)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
print("Random Forest Accuracy:", accuracy_score(y_val, rf_pred))

# Train final model on full data
rf_model.fit(X, y)

# Predict test set
test_predictions = rf_model.predict(test[features])

# Create submission file
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})

submission.to_csv("submission.csv", index=False)
print("submission.csv generated successfully!")