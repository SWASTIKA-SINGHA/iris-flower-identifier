from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    iris = load_iris()
    clf = RandomForestClassifier()
    clf.fit(iris.data, iris.target)
    prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return iris.target_names[prediction[0]]
    def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    import numpy as np
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return model.predict(features)[0]

