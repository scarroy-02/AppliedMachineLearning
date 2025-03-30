from flask import Flask, request, jsonify, render_template
import joblib
from score import score
import os

app = Flask(__name__)

# Load the model data (model and vectorizer)
model_data = joblib.load('saved_models/naive_bayes.joblib')

# Create templates folder if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Create a simple HTML template for the UI
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Spam Classifier</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            min-height: 100px;
            margin-bottom: 10px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
        }
        .spam {
            background-color: #ffdddd;
            border-left: 6px solid #f44336;
        }
        .not-spam {
            background-color: #ddffdd;
            border-left: 6px solid #4CAF50;
        }
        .meter {
            height: 20px;
            background-color: #eee;
            border-radius: 10px;
            position: relative;
            margin-top: 10px;
        }
        .meter-fill {
            height: 100%;
            border-radius: 10px;
            background: linear-gradient(to right, #4CAF50, #f44336);
        }
        .threshold-container {
            margin: 15px 0;
            display: flex;
            align-items: center;
        }
        .threshold-slider {
            flex-grow: 1;
            margin: 0 10px;
        }
        .threshold-value {
            width: 60px;
            text-align: center;
        }
        label {
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Spam Text Classifier</h1>
        <form method="POST" action="/classify">
            <textarea name="text" placeholder="Enter text to classify..." required>{{ text or '' }}</textarea>
            
            <div class="threshold-container">
                <label for="threshold">Threshold:</label>
                <input type="range" id="threshold" name="threshold" class="threshold-slider" 
                       min="0" max="1" step="0.01" value="{{ threshold or 0.5 }}" 
                       oninput="document.getElementById('threshold-value').value = this.value">
                <input type="text" id="threshold-value" class="threshold-value" 
                       value="{{ threshold or 0.5 }}" readonly>
            </div>
            
            <button type="submit">Classify</button>
        </form>
        
        {% if result %}
        <div class="result {% if result.prediction %}spam{% else %}not-spam{% endif %}">
            <h3>Result: {% if result.prediction %}SPAM{% else %}NOT SPAM{% endif %}</h3>
            <p>Propensity: {{ "%.2f"|format(result.propensity * 100) }}%</p>
            <p>Using threshold: {{ threshold }}</p>
            <div class="meter">
                <div class="meter-fill" style="width: {{ result.propensity * 100 }}%"></div>
            </div>
            <p>
                <small>
                    A higher threshold means fewer texts will be classified as spam.
                    <br>A lower threshold means more texts will be classified as spam.
                </small>
            </p>
        </div>
        {% endif %}
    </div>
    
    <script>
        // Allow manual input of threshold value
        document.getElementById('threshold-value').addEventListener('blur', function() {
            var value = parseFloat(this.value);
            if (!isNaN(value) && value >= 0 && value <= 1) {
                document.getElementById('threshold').value = value;
            } else {
                this.value = document.getElementById('threshold').value;
            }
        });
        
        // Make threshold value editable on click
        document.getElementById('threshold-value').addEventListener('click', function() {
            this.readOnly = false;
        });
    </script>
</body>
</html>
    ''')

@app.route('/')
def index():
    """Home page with UI for text classification."""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """Process form submission and classify text."""
    text = request.form.get('text', '')
    
    # Get threshold from form, default to 0.5 if not provided or invalid
    try:
        threshold = float(request.form.get('threshold', 0.5))
        # Ensure threshold is between 0 and 1
        threshold = max(0.0, min(1.0, threshold))
    except ValueError:
        threshold = 0.5
    
    # Classify with the provided threshold
    prediction, propensity = score(text, model_data, threshold)
    
    # Pass the result to the template
    return render_template('index.html', 
                          result={
                              'prediction': prediction,
                              'propensity': propensity
                          },
                          text=text,
                          threshold=threshold)

@app.route('/score', methods=['POST'])
def score_text():
    """
    API endpoint to score a text for spam classification.
    
    Expects JSON with 'text' key and optional 'threshold' key.
    Returns JSON with 'prediction' and 'propensity' keys.
    """
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field in request'}), 400
    
    text = data.get('text', '')
    
    # Get threshold from request, default to 0.5
    try:
        threshold = float(data.get('threshold', 0.5))
        # Ensure threshold is between 0 and 1
        threshold = max(0.0, min(1.0, threshold))
    except (ValueError, TypeError):
        threshold = 0.5
    
    # Classify with the provided threshold
    prediction, propensity = score(text, model_data, threshold)
    
    # Ensure we send proper types
    return jsonify({
        'prediction': bool(prediction),
        'propensity': float(propensity),
        'threshold': threshold
    })

if __name__ == '__main__':
    app.run(debug=True)