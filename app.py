from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Placeholder for models
models = {}

def load_models():
    global models

    # Mock dataset for training
    data = {
        'size': [1500, 1800, 2200, 2400, 2600],  # Example house sizes
        'bedrooms': [3, 4, 3, 4, 5],  # Number of bedrooms
        'bathrooms': [2, 3, 2, 3, 4],  # Number of bathrooms
        'floors': [1, 2, 1, 2, 2],  # Number of floors
        'price': [300000, 350000, 400000, 450000, 500000]  # House price
    }

    df = pd.DataFrame(data)
    
    # Features and target
    X = df[['size', 'bedrooms', 'bathrooms', 'floors']]
    y = df['price']

    # Train models
    models['linear'] = LinearRegression().fit(X, y)
    models['ridge'] = Ridge(alpha=1.0).fit(X, y)
    models['lasso'] = Lasso(alpha=0.1).fit(X, y)
    models['rf'] = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    models['gb'] = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        user_input = request.json
        house_id = user_input.get('houseId', '')
        model_name = user_input.get('model', '')
        features = user_input.get('features', [])
        
        # Ensure features are provided and are a valid list of numbers
        if not features or len(features) == 0:
            return jsonify({'success': False, 'error': 'Features are required.'})

        # Convert the features to a numpy array (2D array for model input)
        features_array = np.array(features).reshape(1, -1)

        # Predict using the selected model
        if model_name in models:
            model = models[model_name]
            prediction = model.predict(features_array)[0]
            return jsonify({'success': True, 'prediction': prediction})
        else:
            return jsonify({'success': False, 'error': 'Model not found'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    load_models()  # Train the models
    app.run(debug=True)
