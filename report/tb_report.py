import numpy as np
from PIL import Image
import datetime
from typing import Tuple, Dict, Optional, List
import cv2
from dataclasses import dataclass
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
        """Initialize report templates and patterns."""
        return {
            "Normal": {
                "findings": {
                    "lung_fields": "Lung fields appear clear without definitive consolidation, infiltrates, or effusions.",
                    "lung_volumes": "Lung volumes appear adequate with visible costophrenic angles.",
                    "cardiovascular": "Heart size and mediastinal contours appear within normal limits.",
                    "pleural_space": "No definitive evidence of pleural effusion or pneumothorax.",
                    "bones": "No acute osseous abnormalities identified.",
                    "soft_tissues": "Soft tissues appear unremarkable.",
                },
                "impression": "No definitive acute cardiopulmonary findings identified on this examination.",
                "recommendations": [
                    "Clinical correlation is recommended",
                    "Consider follow-up imaging if symptoms persist or worsen",
                    "Compare with prior studies if available",
                ],
            },
            "Tuberculosis": {
                "findings": {
                    "lung_opacity": {
                        "severe": "Areas of increased opacity noted in the lung fields, potentially representing severe consolidation patterns consistent with tuberculosis.",
                        "moderate": "Patchy areas of increased opacity noted, possibly representing moderate consolidation suggestive of tuberculosis.",
                        "mild": "Subtle areas of increased opacity noted, may represent early or mild tuberculosis manifestation.",
                    },
                    "cardiovascular": {
                        "normal": "Cardiac silhouette appears within normal limits. Mediastinal contours are preserved.",
                        "abnormal": "Cardiac silhouette appears mildly enlarged. Further evaluation of mediastinal structures may be warranted.",
                    },
                    "pleural_space": {
                        "severe": "Possible significant pleural effusion noted with blunting of costophrenic angles.",
                        "moderate": "Possible moderate pleural effusion with blunting of costophrenic angles.",
                        "mild": "Minimal blunting of costophrenic angles noted, may represent small pleural effusion.",
                        "normal": "Costophrenic angles appear preserved.",
                    },
                },
                "severity_recommendations": {
                    "severe": [
                        "Clinical correlation strongly recommended",
                        "Consider additional imaging studies for confirmation",
                        "Monitor clinical status closely",
                        "Consider infectious disease consultation",
                        "Follow-up imaging recommended based on clinical course",
                    ],
                    "moderate": [
                        "Clinical correlation recommended",
                        "Consider follow-up imaging in 24-48 hours if symptoms persist",
                        "Monitor for clinical improvement",
                        "Consider additional diagnostic testing if clinically indicated",
                        "Compare with prior studies if available",
                    ],
                    "mild": [
                        "Clinical correlation recommended",
                        "Consider follow-up imaging if symptoms worsen",
                        "Monitor clinical course",
                        "Compare with prior studies if available",
                        "Consider additional views if clinically indicated",
                    ],
                },
            },
            "Uncertain": {
                "findings": "The radiographic findings are indeterminate. Technical factors or patient positioning may limit interpretation.",
                "impression": "Findings are inconclusive and require clinical correlation and possibly additional imaging.",
                "recommendations": [
                    "Clinical correlation is essential",
                    "Consider additional views or imaging modalities",
                    "Compare with prior studies if available",
                    "Follow-up imaging may be warranted based on clinical presentation",
                ],
            },
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
            "condition": classification,
            "confidence_metrics": {
                "mean_prediction": float(mean_pred),
                "std_prediction": float(std_pred),
            },
        }
        return classification, float(confidence), analysis

    def _determine_severity(self, confidence: float) -> str:
        """Determine the severity level based on confidence score."""
        if confidence > 0.9:
            return "severe"
        elif confidence > 0.75:
            return "moderate"
        else:
            return "mild"

    def generate_report(
        self, image_path: str, patient_info: PatientInfo = PatientInfo()
    ) -> Dict:
        classification, confidence, analysis = self.analyze_image(image_path)
        
        # Get the appropriate report template
        template = self.report_patterns[classification]
        
        # Determine severity for tuberculosis cases
        severity = self._determine_severity(confidence) if classification == "Tuberculosis" else None

        report = {
            "patient_info": {
                "name": patient_info.name or "Not Provided",
                "age": patient_info.age or "Not Provided",
                "gender": patient_info.gender or "Not Provided",
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "referring_physician": patient_info.referring_physician or "Not Provided",
                "medical_history": patient_info.medical_history or "Not Provided",
                "symptoms": patient_info.symptoms or [],
            },
            "study": {
                "type": "Chest X-ray",
                "view": "PA",
                "reason_for_examination": "Suspected Tuberculosis",
                "image_quality": "Adequate" if classification != "Uncertain" else "Limited",
            },
            "analysis_metrics": analysis["confidence_metrics"],
        }

        # Add findings based on classification
        if classification == "Normal":
            report.update({
                "findings": template["findings"],
                "impression": template["impression"],
                "recommendations": template["recommendations"]
            })
        elif classification == "Tuberculosis":
            report.update({
                "findings": {
                    "lung_opacity": template["findings"]["lung_opacity"][severity],
                    "cardiovascular": template["findings"]["cardiovascular"]["normal"],
                    "pleural_space": template["findings"]["pleural_space"][severity],
                },
                "impression": f"Findings suggestive of {severity} tuberculosis with {confidence:.1%} confidence.",
                "recommendations": template["severity_recommendations"][severity]
            })
        else:  # Uncertain
            report.update({
                "findings": template["findings"],
                "impression": template["impression"],
                "recommendations": template["recommendations"]
            })

        return report