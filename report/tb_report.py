import numpy as np
from PIL import Image
import datetime
from typing import Tuple, Dict
import cv2
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import tensorflow as tf


@dataclass
class PatientInfo:
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    referring_physician: Optional[str] = None
    medical_history: Optional[str] = None
    symptoms: Optional[List[str]] = None


class TBAnalysisModel:
    def __init__(self, model_path: str, confidence_threshold: float = 0.75):
        """Initialize the report generator with a trained TensorFlow model."""
        self.model = tf.keras.models.load_model(model_path)
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.confidence_threshold = confidence_threshold
        self.report_patterns = self._initialize_report_patterns()

    def _initialize_report_patterns(self) -> Dict:
        """Initialize standard reporting patterns and templates."""
        return {
            "normal": {
                "description": "No significant abnormalities detected",
                "recommendations": "Routine follow-up as clinically indicated"
            },
            "tuberculosis": {
                "description": "Findings suggestive of tuberculosis",
                "recommendations": "Clinical correlation and further diagnostic testing recommended"
            },
            "uncertain": {
                "description": "Findings are inconclusive",
                "recommendations": "Consider repeat imaging or alternative diagnostic methods"
            }
        }

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, bool]:
        try:
            target_size = (224, 224)
            img = Image.open(image_path).convert("RGB")

            if img.size[0] < 200 or img.size[1] < 200:
                return None, False

            img_array = np.array(img)
            mean_intensity = np.mean(img_array)
            std_intensity = np.std(img_array)

            if mean_intensity < 30 or mean_intensity > 225 or std_intensity < 10:
                return None, False

            img = img.resize(target_size)
            img_array = np.array(img) / 255.0
            return np.expand_dims(img_array, axis=0), True

        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None, False

    def analyze_image(self, image_path: str) -> Tuple[str, float, Dict]:
        processed_img, is_valid = self.preprocess_image(image_path)
        if not is_valid:
            return "Uncertain", 0.0, {"error": "Invalid or poor quality image"}

        predictions = [
            self.model.predict(processed_img, verbose=0)[0][0] for _ in range(5)
        ]
        mean_pred, std_pred = np.mean(predictions), np.std(predictions)

        if std_pred > 0.15:
            return (
                "Uncertain",
                float(mean_pred),
                {"uncertainty": "High prediction variance"},
            )

        if mean_pred > self.confidence_threshold:
            classification = "Normal"
            confidence = 1 - mean_pred
        elif mean_pred < (1 - self.confidence_threshold):
            classification = "Tuberculosis"
            confidence = mean_pred
        else:
            return "Uncertain", float(mean_pred), {"uncertainty": "Low confidence"}

        analysis = {
            "condition": (
                "Normal"
                if classification == "Normal"
                else "Findings suggestive of tuberculosis"
            ),
            "confidence_metrics": {
                "mean_prediction": float(mean_pred),
                "std_prediction": float(std_pred),
            },
        }
        return classification, float(confidence), analysis

    def generate_tb_report(
        self, image_path: str, patient_info: PatientInfo = PatientInfo()
    ) -> Dict:
        classification, confidence, analysis = self.analyze_image(image_path)
        diagnosis = "Tuberculosis" if classification == "Tuberculosis" else "Normal"

        report = {
            "patient_info": {
                "name": patient_info.name or "Not Provided",
                "age": patient_info.age or "Not Provided",
                "gender": patient_info.gender or "Not Provided",
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "referring_physician": patient_info.referring_physician
                or "Not Provided",
            },
            "study": {
                "type": "Chest X-ray",
                "view": "PA",
                "reason_for_examination": "Suspected Tuberculosis",
            },
            "findings": {
                "lung_fields": {
                    "upper_lobe_opacities": diagnosis == "Tuberculosis",
                    "infiltrates_or_nodules": diagnosis == "Tuberculosis",
                    "cavitation_or_fibrosis": diagnosis == "Tuberculosis",
                    "volume_loss": diagnosis == "Tuberculosis",
                    "interstitial_pattern": diagnosis == "Tuberculosis",
                },
            },
            "impression": {
                "summary_of_findings": f"There is {'evidence of tuberculosis' if diagnosis == 'Tuberculosis' else 'no evidence of tuberculosis'}.",
                "diagnosis": diagnosis,
                "acute_pathology": (
                    "None" if diagnosis == "Normal" else "Possible tuberculosis"
                ),
            },
            "recommendations": {
                "clinical_correlation": (
                    "Further clinical tests suggested"
                    if diagnosis == "Tuberculosis"
                    else "Routine follow-up"
                ),
                "follow_up": "6 months" if diagnosis == "Normal" else "Immediate",
            },
        }

        return report