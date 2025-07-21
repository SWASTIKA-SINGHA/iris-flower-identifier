import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to train the model and return it along with iris data
def train_model():
    # Load the Iris dataset
    iris = load_iris()

    # Convert to DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].apply(lambda x: iris.target_names[x])

    # Visualization (optional, not used in app.py)
    # sns.pairplot(df, hue='species_name')
    # plt.show()

    # Prepare input and output
    X = iris.data
    y = iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Evaluate (optional - for your testing, not needed in app)
    y_pred = model.predict(X_test)
    print("✅ Accuracy of model:", accuracy_score(y_test, y_pred))
    print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

    return model, iris

# Function to predict flower type based on features
def predict_flower(model, iris, features):
    prediction = model.predict([features])
    return iris.target_names[prediction[0]]

# Optional: only runs when you execute project_py.py directly
if __name__ == "__main__":
    model, iris = train_model()
    # Example test case
    sepal_length = 5.1
    sepal_width = 3.5
    petal_length = 1.4
    petal_width = 0.2
    result = predict_flower(model, iris, [sepal_length, sepal_width, petal_length, petal_width])
    print("🌸 The flower is:", result)
