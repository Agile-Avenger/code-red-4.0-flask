# models.py
from typing import Optional, List

# Define the schema for PatientInfo
patient_info_schema = {
    "name": str,
    "age": int,
    "gender": str,
    "referring_physician": str,
    "medical_history": str,
    "symptoms": list,
}


def validate_document(data, schema):
    """Validates the document against the provided schema."""
    for field, field_type in schema.items():
        if field in data and data[field] is not None:
            if not isinstance(data[field], field_type):
                return False, f"Field '{field}' must be of type {field_type.__name__}"
        elif schema[field] is not Optional:
            return False, f"Field '{field}' is required and cannot be None."
    return True, ""
