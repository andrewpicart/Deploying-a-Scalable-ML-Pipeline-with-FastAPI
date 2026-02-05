import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference

# TODO: implement the first test. Change the function name and input as needed
@pytest.fixture
def data():
    """
    Creates a simple dataset for testing.
    """
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    return X, y

def test_train_model(data):
    """
    Test that the train_model function returns a RandomForestClassifier.
    """
    X, y = data
    model = train_model(X, y)
    
    # Check if the model is of the expected type
    assert isinstance(model, RandomForestClassifier)
    
    # Check if the model is actually fitted (has the classes_ attribute)
    assert hasattr(model, "classes_")


def test_inference(data):
    """
    Test that inference returns predictions of the correct shape and type.
    """
    X, y = data
    model = train_model(X, y)
    preds = inference(model, X)
    
    # Check that predictions is a numpy array
    assert isinstance(preds, np.ndarray)
    
    # Check that we get one prediction per input row
    assert len(preds) == len(X)


def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns three float values (precision, recall, fbeta).
    """
    # Create dummy actuals and predictions
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Check types
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    
    # Check ranges (metrics should be between 0 and 1)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
