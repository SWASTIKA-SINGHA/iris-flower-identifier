# project_py.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Train the model once globally
iris = load_iris()
X = iris.data
y = iris.target
model = RandomForestClassifier()
model.fit(X, y)

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    return iris.target_names[prediction[0]]
