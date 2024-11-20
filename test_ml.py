import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

# implement the first test
def test_train_model():
    """
    # Test if the train_model function returns a RandomForestClasifier
    """
    # a dummy dataset
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = train_model(X, y)
    
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"


# implement the second test. 
def test_compute_model_metrics():
    """
    # Test if compute_model_metrics function returns expected type of result and values
    """
    y = np.array([0, 1, 1, 0, 1])
    preds = np.array([0, 1, 0, 0, 1])
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    # check if the returned values are floats 
    # and check if the values are within the expected range

    assert isinstance(precision, float), "Precision is not a float"
    assert isinstance(recall, float), "Recall is not a float"
    assert isinstance(fbeta, float), "Fbeta is not a float"
    assert 0 <= precision <= 1, "Precision is not between 0 and 1"
    assert 0 <= recall <= 1, "Recall is not between 0 and 1"
    assert 0 <= fbeta <= 1, "Fbeta is not between 0 and 1"


# implement the third test.
def test_inference():
    """
    # Test if the inference function returns predictions of the expected shape and type
    """
    
    # create a dummy model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # create dummy data
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    
    # fitting the model
    model.fit(X_train, y_train)
    
    # dummy test data
    X_test = np.random.rand(20, 5)
    
    predictions = inference(model, X_test)
    
    # check if predictions have the expected shape and type
    assert isinstance(predictions, np.ndarray), "Predictions must be a numpy array"
    assert predictions.shape == (20,), f"expected shape is (20,), got {predictions.shape}"
    assert np.isin(predictions, [0, 1]).all(), "Predictions should be binary (0 or 1)"