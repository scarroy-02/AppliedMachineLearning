import joblib
import sklearn
import numpy as np

def score(text: str, 
          model: dict, 
          threshold: float) -> tuple[bool, float]:
    """
    Score a text using a trained machine learning model.
    
    Args:
        text (str): Input text to classify
        model (dict): Dictionary containing the trained model and vectorizer
                     {'model': trained_model, 'vectorizer': vectorizer}
        threshold (float): Probability threshold for positive classification
    
    Returns:
        tuple[bool, float]: 
            - First element: Boolean prediction (True if spam, False if not)
            - Second element: Propensity score (probability of being spam)
    """
    # Extract model and vectorizer
    classifier = model['model']
    vectorizer = model['vectorizer']
    
    # Vectorize the input text
    text_vectorized = vectorizer.transform([text])
    
    # Get probability and make prediction
    propensity = float(classifier.predict_proba(text_vectorized)[0][1])
    prediction = bool(propensity >= threshold)  # Convert to Python bool explicitly
    
    return prediction, propensity