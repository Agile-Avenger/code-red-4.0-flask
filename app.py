# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import tempfile
from flask_pymongo import PyMongo
from firebase_admin import auth, credentials, initialize_app

from report.pneumonia_report import XRayReportGenerator, PatientInfo
from report.tb_report import TBAnalysisModel
from mongo import mongo

app = Flask(__name__)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/medidignose"
mongo = PyMongo(app)

# Initialize Firebase Admin SDK
cred = credentials.Certificate(
    "./secrets/medidignose-firebase-adminsdk-bpud2-d6cdfa06f0.json"
)
initialize_app(cred)


# Middleware to verify Firebase Auth token and extract UID
def verify_firebase_token():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None, "Authorization token is missing or invalid"
    token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token["uid"]
        return uid, None
    except Exception as e:
        return None, str(e)


@app.route("/add_patient", methods=["POST"])
def add_patient():
    uid, error = verify_firebase_token()
    if error:
        return jsonify(error=error), 401

    data = request.json
    data["uid"] = uid  # Associate the data with the UID
    is_valid, message = mongo.validate_document(data, mongo.patient_info_schema)
    if is_valid:
        mongo.db.patients.update_one({"uid": uid}, {"$set": data}, upsert=True)
        return jsonify(message="Patient record added/updated successfully"), 201
    else:
        return jsonify(error=message), 400


@app.route("/patient", methods=["GET"])
def get_patient():
    uid, error = verify_firebase_token()
    if error:
        return jsonify(error=error), 401

    patient = mongo.db.patients.find_one({"uid": uid})
    if patient:
        patient_data = {
            field: patient.get(field) for field in mongo.patient_info_schema.keys()
        }
        patient_data["uid"] = uid
        return jsonify(patient_data), 200
    else:
        return jsonify(error="Patient not found"), 404


@app.route("/update_patient", methods=["PUT"])
def update_patient():
    uid, error = verify_firebase_token()
    if error:
        return jsonify(error=error), 401

    data = request.json
    is_valid, message = mongo.validate_document(data, mongo.patient_info_schema)
    if is_valid:
        result = mongo.db.patients.update_one({"uid": uid}, {"$set": data})
        if result.modified_count:
            return jsonify(message="Patient record updated successfully"), 200
        else:
            return jsonify(error="Patient not found or no changes made"), 404
    else:
        return jsonify(error=message), 400


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


@app.route("/generate-pneumonia-report", methods=["POST"])
def create_pneumonia_report():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        # Create a temporary file to save the uploaded image
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "temp_xray.png")

        # Save the uploaded file to temporary location
        image = Image.open(io.BytesIO(file.read()))
        image.save(temp_path)

        # Initialize the generator
        generator = XRayReportGenerator(model_path="./models/pneumonia.h5")

        # Create patient info
        patient_info = PatientInfo(
            name=request.form.get("name", "Not Provided"),
            age=int(request.form.get("age", 0)) if request.form.get("age") else None,
            gender=request.form.get("gender", "Not Provided"),
            referring_physician=request.form.get("referring_physician", "Not Provided"),
            medical_history=request.form.get("medical_history", "Not Provided"),
            symptoms=(
                request.form.get("symptoms", "").split(",")
                if request.form.get("symptoms")
                else None
            ),
        )

        # Generate the report using the temporary file path
        report = generator.generate_pneumonia_report(
            image_path=temp_path, patient_info=patient_info
        )

        # Clean up the temporary file
        try:
            os.remove(temp_path)
        except:
            pass  # Ignore cleanup errors

        return jsonify({"report": report})

    except Exception as e:
        # Log the full error for debugging
        print(f"Error generating report: {str(e)}")
        return jsonify({"error": str(e)}), 400


@app.route("/generate-tb-report", methods=["POST"])
def create_tb_report():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        # Create a temporary file to save the uploaded image
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "temp_xray.png")

        # Save the uploaded file to temporary location
        image = Image.open(io.BytesIO(file.read()))
        image.save(temp_path)

        # Initialize the generator
        generator = TBAnalysisModel(model_path="./models/tb.h5")

        # Create patient info
        patient_info = PatientInfo(
            name=request.form.get("name", "Not Provided"),
            age=int(request.form.get("age", 0)) if request.form.get("age") else None,
            gender=request.form.get("gender", "Not Provided"),
            referring_physician=request.form.get("referring_physician", "Not Provided"),
            medical_history=request.form.get("medical_history", "Not Provided"),
            symptoms=(
                request.form.get("symptoms", "").split(",")
                if request.form.get("symptoms")
                else None
            ),
        )

        # Generate the report using the temporary file path
        report = generator.generate_tb_report(
            image_path=temp_path, patient_info=patient_info
        )

        # Clean up the temporary file
        try:
            os.remove(temp_path)
        except:
            pass  # Ignore cleanup errors

        return jsonify({"report": report})

    except Exception as e:
        # Log the full error for debugging
        print(f"Error generating report: {str(e)}")
        return jsonify({"error": str(e)}), 400


@app.route("/predict_pneumonia", methods=["POST"])
def predict_pneumonia():
    return predict_from_model(pneumonia_model, target_size=(224, 224))


@app.route("/predict_tb", methods=["POST"])
def predict_tb():
    return predict_from_model(tb, target_size=(224, 224))


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
