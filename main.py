"""
End-to-end Machine Learning pipeline using Logistic Regression
on the Iris dataset with proper structure and evaluation.
"""

from typing import Tuple
import logging

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_data() -> Tuple:
    """
    Load dataset and split into training and testing sets.

    Returns:
        Tuple containing x_train, x_test, y_train, y_test
    """
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42
    )
    logging.info("Data loaded and split successfully")
    return x_train, x_test, y_train, y_test


def create_model() -> LogisticRegression:
    """
    Initialize the Logistic Regression model.

    Returns:
        LogisticRegression model
    """
    model = LogisticRegression(max_iter=200)
    logging.info("Model created")
    return model


def train_model(model: LogisticRegression, x_train, y_train) -> LogisticRegression:
    """
    Train the model with training data.

    Args:
        model: ML model
        x_train: Training features
        y_train: Training labels

    Returns:
        Trained model
    """
    model.fit(x_train, y_train)
    logging.info("Model training completed")
    return model


def evaluate_model(model: LogisticRegression, x_test, y_test) -> float:
    """
    Evaluate the model performance.

    Args:
        model: Trained model
        x_test: Test features
        y_test: Test labels

    Returns:
        Accuracy score
    """
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    logging.info("Model evaluation completed")
    print(f"Model Accuracy: {accuracy:.2f}")
    return accuracy


def main() -> None:
    """Main execution pipeline."""
    try:
        x_train, x_test, y_train, y_test = load_data()
        model = create_model()
        trained_model = train_model(model, x_train, y_train)
        evaluate_model(trained_model, x_test, y_test)
    except Exception as error:
        logging.error("An error occurred: %s", error)


if __name__ == "__main__":
    main()