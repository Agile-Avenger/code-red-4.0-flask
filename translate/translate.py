import json
from google.cloud import translate
from datetime import datetime
from typing import Union, Dict, List, Optional
import os


class MedicalReportTranslator:
    def __init__(self, credentials_path: str):
        """Initialize the translator with Google Cloud credentials"""
        self.client = translate.TranslationServiceClient.from_service_account_json(credentials_path)
        self.parent = None

    def set_project(self, project_id: str):
        """Set the Google Cloud project ID"""
        self.parent = f"projects/{project_id}/locations/global"

    def translate_text(self, text: str, target_language: str) -> str:
        """Translate a single text string"""
        if not self.parent:
            raise ValueError("Project ID not set. Call set_project() first.")

        response = self.client.translate_text(
            request={
                "parent": self.parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": "en",
                "target_language_code": target_language,
            }
        )
        return response.translations[0].translated_text

    def _should_translate_key(self, key: str) -> bool:
        """Determine if a key's value should be translated based on its name"""
        non_translatable_keys = {
            "date", "age", "gender", "type", "view", "image_quality", "name",
            "referring_physician", "analysis_metrics", "confidence_metrics"
        }
        return key not in non_translatable_keys

    def translate_value(self, value: Union[str, Dict, List], target_language: str, current_key: str = None) -> Union[str, Dict, List]:
        """Recursively translate a value that could be a string, dictionary, or list"""
        if isinstance(value, str):
            if current_key and not self._should_translate_key(current_key):
                return value
            if value.strip() and not value.isdigit():
                return self.translate_text(value, target_language)
            return value
        elif isinstance(value, dict):
            return {k: self.translate_value(v, target_language, k) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.translate_value(item, target_language) for item in value]
        else:
            return value

    def _handle_classification_specific_content(
        self,
        content: Dict,
        classification: str,
        severity: str,
        template: Dict,
        target_language: str,
    ) -> Dict:
        """Handle translation of classification-specific content"""
        translated_content = {}

        if classification == "Normal":
            translated_findings = self.translate_value(
                template["Normal"]["findings"], target_language
            )
            translated_content = {
                "findings": translated_findings,
                "impression": self.translate_value(
                    template["Normal"]["impression"], target_language
                ),
                "recommendations": self.translate_value(
                    template["Normal"]["recommendations"], target_language
                ),
            }
        elif classification == "Tuberculosis":
            findings = {
                "lung_opacity": self.translate_value(
                    template["Tuberculosis"]["findings"]["lung_opacity"][severity],
                    target_language
                ),
                "cardiovascular": self.translate_value(
                    template["Tuberculosis"]["findings"]["cardiovascular"]["normal"],
                    target_language
                ),
                "pleural_space": self.translate_value(
                    template["Tuberculosis"]["findings"]["pleural_space"][severity],
                    target_language
                ),
            }
            translated_content = {
                "findings": findings,
                "impression": self.translate_value(
                    f"Findings suggestive of {severity} tuberculosis", target_language
                ),
                "recommendations": self.translate_value(
                    template["Tuberculosis"]["severity_recommendations"][severity],
                    target_language
                ),
            }
        else:  # Uncertain
            template_uncertain = template.get("Uncertain", {
                "findings": "Further examination needed",
                "impression": "Findings are uncertain",
                "recommendations": ["Clinical correlation and follow-up recommended"]
            })
            translated_content = {
                "findings": self.translate_value(
                    template_uncertain["findings"], target_language
                ),
                "impression": self.translate_value(
                    template_uncertain["impression"], target_language
                ),
                "recommendations": self.translate_value(
                    template_uncertain["recommendations"], target_language
                ),
            }

        return translated_content

    def translate_medical_report(
        self,
        report: Dict,
        template: Dict,
        classification: str,
        severity: str = None,
        target_language: str = "es",
    ) -> Dict:
        """
        Translate a medical report with its associated template
        """
        try:
            if classification not in ["Normal", "Tuberculosis", "Uncertain"]:
                raise ValueError(f"Invalid classification: {classification}")
            
            if classification == "Tuberculosis" and not severity:
                raise ValueError("Severity is required for Tuberculosis classification")

            translated_report = {
                "patient_info": self.translate_value(
                    report["patient_info"], target_language
                ),
                "study": self.translate_value(report["study"], target_language),
            }

            if "analysis_metrics" in report:
                translated_report["analysis_metrics"] = report["analysis_metrics"]

            classification_content = self._handle_classification_specific_content(
                report, classification, severity, template, target_language
            )
            translated_report.update(classification_content)

            return translated_report

        except Exception as e:
            raise Exception(f"Translation error: {str(e)}")



    # Set the path to your credentials file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    credentials_path = os.path.join(current_dir, "./secrets/medi-dignose-8001634df36a.json")  # Put your JSON file in the same directory
    project_id = "medi-dignose"  # Replace with your project ID

    # Sample template data
    template_data = {
        "Normal": {
            "findings": {
                "lung_fields": "Lung fields appear clear without definitive consolidation.",
                "cardiovascular": "Heart size appears normal.",
            },
            "impression": "No acute findings.",
            "recommendations": ["Clinical correlation recommended"],
        },
        "Tuberculosis": {
            "findings": {
                "lung_opacity": {
                    "severe": "Areas of increased opacity noted in lung fields."
                },
                "cardiovascular": {"normal": "Cardiac silhouette appears normal."},
                "pleural_space": {"severe": "Significant pleural effusion noted."},
            },
            "severity_recommendations": {
                "severe": ["Urgent clinical correlation recommended"]
            },
        },
    }

    # Sample report data
    sample_report = {
        "patient_info": {
            "name": "Not Provided",
            "age": "45",
            "gender": "Male",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "medical_history": "History of respiratory issues",
            "symptoms": ["Persistent cough", "Fever"],
        },
        "study": {
            "type": "Chest X-ray",
            "view": "PA",
            "reason_for_examination": "Suspected Tuberculosis",
            "image_quality": "Adequate",
        },
    }

    try:
        # Initialize translator with credentials file
        translator = MedicalReportTranslator(credentials_path)
        translator.set_project(project_id)
        
        translated_report = translator.translate_medical_report(
            sample_report,
            template_data,
            classification="Tuberculosis",
            severity="severe",
            target_language="hi",
        )


        print(json.dumps(translated_report, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Error: {str(e)}")