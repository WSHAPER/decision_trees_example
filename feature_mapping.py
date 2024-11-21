"""
This module contains the feature name mapping for the heart disease dataset.
It provides human-readable names for features used in visualization and evaluation.
"""

FEATURE_NAMES = {
    'age': 'Age',
    'sex': 'Gender',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Cholesterol Level',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG Results',
    'thalach': 'Maximum Heart Rate',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression',
    'slope': 'ST Slope',
    'ca': 'Number of Major Vessels',
    'thal': 'Thalassemia Type'
}

# Additional mappings for categorical variables
CHEST_PAIN_TYPES = {
    0: 'Typical Angina',
    1: 'Atypical Angina',
    2: 'Non-Anginal Pain',
    3: 'Asymptomatic'
}

GENDER_MAP = {
    0: 'Female',
    1: 'Male'
}

ECG_RESULTS = {
    0: 'Normal',
    1: 'ST-T Wave Abnormality',
    2: 'Left Ventricular Hypertrophy'
}

SLOPE_TYPES = {
    0: 'Upsloping',
    1: 'Flat',
    2: 'Downsloping'
}

THALASSEMIA_TYPES = {
    0: 'Normal',
    1: 'Fixed Defect',
    2: 'Reversible Defect',
    3: 'Unknown'
}

def get_feature_display_name(feature_name):
    """Get the human-readable display name for a feature."""
    return FEATURE_NAMES.get(feature_name, feature_name)

def get_all_feature_names():
    """Get a list of all feature names in human-readable format."""
    return list(FEATURE_NAMES.values())
