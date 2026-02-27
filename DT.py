# DT.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X = [[30],[40],[50],[60],[20],[10],[70]]
y = [0,1,1,1,0,0,1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction - DON'T redefine X_test, use the one from train_test_split
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# If you want to test a single value separately:
single_prediction = model.predict([[60]])

print("Prediction for [60]:", single_prediction)
