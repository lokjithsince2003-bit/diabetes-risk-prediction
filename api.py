from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# load saved model
model = joblib.load("model/diabetes_model.pkl")

@app.route("/")
def home():
    return "Diabetes Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])[0]

    if prediction == 1:
        result = "Diabetes Risk"
    else:
        result = "No Diabetes Risk"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)