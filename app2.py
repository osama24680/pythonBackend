import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)
model = pickle.load(open("calories_model.pkl", "rb"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        age = data.get("age")
        duration = data.get("duration")
        temperature = data.get("temperature")
        heart_rate = data.get("heart_rate")

        input_features = np.array(
            [
                [
                    age,
                    duration,
                    temperature,
                    heart_rate,
                ]
            ]
        )
        prediction = model.predict(input_features)

        return jsonify({"calories": round(prediction[0], 1)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
