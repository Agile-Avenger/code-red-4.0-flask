# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load each model
pneumonia_model = tf.keras.models.load_model("models/pneumonia.h5")
tb = tf.keras.models.load_model("models/tb.h5")


def preprocess_image(image, target_size):
    # Ensure the image is in RGB mode
    image = image.convert("RGB")
    # Resize and normalize the image
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize pixel values if required by the model
    image = np.expand_dims(
        image, axis=0
    )  # Add batch dimension: shape becomes (1, 224, 224, 3)
    return image


@app.route("/predict_pneumonia", methods=["POST"])
def predict_pneumonia():
    return predict_from_model(pneumonia_model, target_size=(224, 224))


@app.route("/predict_tb", methods=["POST"])
def predict_tb():
    return predict_from_model(
        tb, target_size=(224, 224)
    )  # Adjust target size if different


def predict_from_model(model, target_size):
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image to fit the model input shape
        input_data = preprocess_image(image, target_size)

        # Make prediction
        prediction = model.predict(input_data)

        # Convert prediction to list if it's an array
        prediction = prediction.tolist()

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
