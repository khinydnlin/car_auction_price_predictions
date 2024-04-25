from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask('car auction price predictions')

# Load the trained model
model_pipeline = joblib.load('model_pipeline.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from POST request in JSON format
        data = {
            'car_brand': request.form['car_brand'],
            'car_make': request.form['car_make'],
            'body_type': request.form['body_type'],
            'year': int(request.form['year']),
            'engine_power':request.form['engine_power'],
            'mileage': int(request.form['mileage']),
            'transmission': request.form['transmission'],
            
        }
        # Create a DataFrame from the JSON data
        input_df = pd.DataFrame([data])  # Encapsulate data in a list to ensure it's treated as a single row
        # Generate predictions
        predictions = model_pipeline.predict(input_df)
        # Round predictions
        rounded_predictions = [round(pred) for pred in predictions]
        # Return predictions as JSON
        return jsonify(prediction=rounded_predictions[0])
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
