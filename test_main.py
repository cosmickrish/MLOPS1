from main import model, X_test, y_test
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.9  # Example assertion
