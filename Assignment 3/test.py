import joblib
import requests
import time
import subprocess
from score import score

# Load the model and vectorizer
model_data = joblib.load('saved_models/naive_bayes.joblib')

def test_score_function():
    """
    Comprehensive unit test for the score function.
    
    Tests multiple scenarios to validate scoring function behavior.
    """
    # Smoke test
    result = score("test text", model_data, 0.5)
    assert result is not None, "Function should return a result"
    
    # Format test
    prediction, propensity = result
    assert isinstance(prediction, bool), "Prediction should be a boolean"
    assert isinstance(propensity, float), "Propensity should be a float"
    
    # Threshold tests
    zero_threshold_result = score("test text", model_data, 0.0)
    assert zero_threshold_result[0] is True, "Zero threshold should always predict True"
    
    one_threshold_result = score("test text", model_data, 1.0)
    assert one_threshold_result[0] is False, "One threshold should always predict False"
    
    # Propensity score range test
    assert 0 <= propensity <= 1, "Propensity score should be between 0 and 1"
    
    # Spam and non-spam tests
    spam_text = "FREE CAR CLICK HERE NOW!!! LIMITED OFFER!!!"
    spam_prediction, spam_propensity = score(spam_text, model_data, 0.5)
    assert spam_prediction is True, "Obvious spam text should be classified as spam"
    
    non_spam_text = "Hi, let's discuss the project details tomorrow."
    non_spam_prediction, non_spam_propensity = score(non_spam_text, model_data, 0.5)
    assert non_spam_prediction is False, "Non-spam text should not be classified as spam"

def test_flask_endpoint():
    """
    Integration test for Flask scoring endpoint.
    
    Launches the Flask app, tests the endpoint, and closes the app.
    """
    # Launch Flask app
    flask_process = subprocess.Popen(['flask', '--app', 'app', 'run'])
    
    try:
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Test endpoint
        test_text = "FREE CAR CLICK HERE NOW!!!"
        response = requests.post('http://localhost:5000/score', json={'text': test_text})
        
        # Validate response
        assert response.status_code == 200, "Endpoint should return 200 OK"
        
        result = response.json()
        assert 'prediction' in result, "Response should contain prediction"
        assert 'propensity' in result, "Response should contain propensity"
        
        # Accept either boolean or integer (0/1) for prediction
        assert result['prediction'] in [True, False, 0, 1], "Prediction should be boolean or 0/1"
        assert 0 <= result['propensity'] <= 1, "Propensity should be between 0 and 1"
        
        # Test with non-spam text
        non_spam_text = "Meeting scheduled for tomorrow at 10 AM"
        non_spam_response = requests.post('http://localhost:5000/score', 
                                          json={'text': non_spam_text})
        
        non_spam_result = non_spam_response.json()
        assert non_spam_result['prediction'] in [False, 0], "Non-spam should be False or 0"
    
    finally:
        # Terminate Flask app
        flask_process.terminate()
        flask_process.wait()