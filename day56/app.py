from flask import Flask, request, jsonify
import joblib

# Create Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("iris_model.pkl")

# Home route
@app.route("/")
def home():
    return "ML Model Deployment API Running Successfully!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    # Get JSON data
    data = request.json["features"]

    # Make prediction
    prediction = model.predict([data])

    # Return response
    return jsonify({
        "prediction": int(prediction[0])
    })

# Run app
if __name__ == "__main__":
    app.run(debug=True)