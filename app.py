from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('xgboost_regressor.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')  # Renders the HTML page


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get feature data from the JSON body
        data = request.get_json()
        features = data['features']

        # Ensure that the features are in the correct format and scale them
        features_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_value = float(prediction[0])

        # Define the threshold for authorization
        threshold = 0.5  # Example threshold (adjust as per your use case)
        status = "authorized" if prediction_value > threshold else "not authorized"

        # Return the prediction and authorization status as a JSON response
        return jsonify({"prediction": prediction_value, "status": status})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
