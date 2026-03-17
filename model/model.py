import os
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# Setup paths relative to the script location for portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")


def random_forest(data):
    """
    Trains a Random Forest Classifier on the provided dataset.

    Splits the data into training and testing sets, fits the model,
    and prints evaluation metrics (Accuracy and Confusion Matrix).

    Args:
        data (pd.DataFrame): The cleaned diabetes dataset.

    Returns:
        RandomForestClassifier: The trained Scikit-Learn model object.
    """
    X = data.drop(["Diabetes_binary"], axis=1)
    y = data["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return model


def make_prediction(model, X_input):
    """
    Generates predictions using the trained model for a given set of input features.

    Args:
        model: Trained Scikit-Learn classifier.
        X_input (array-like): Feature matrix (21 features) for prediction.

    Returns:
        np.array: Predicted classes (0 or 1).
    """
    prediction = model.predict(X_input)
    return prediction


def clean_data():
    """
    Loads raw data from CSV, removes duplicates, and ensures all features
    are in integer format for the model.

    Returns:
        pd.DataFrame: A cleaned and deduplicated DataFrame.
    """
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates()
    data = data.astype(int)
    return data


def main():
    """
    Main execution pipeline:
    1. Loads and cleans data.
    2. Trains the Random Forest model.
    3. Serializes the model into a pickle file for deployment.
    """
    data = clean_data()
    model = random_forest(data)

    with open("model.pkl", "wb") as f:
        pkl.dump(model, f)


if __name__ == '__main__':
    main()
