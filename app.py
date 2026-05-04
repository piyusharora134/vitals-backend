from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)
CORS(app)

# 🔥 Lazy loading (IMPORTANT FIX)
model = None

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = joblib.load("model.pkl")
        fix_multi_class(model)
    return model

# Fix for sklearn 1.6+ compatibility
def fix_multi_class(obj):
    if hasattr(obj, 'multi_class'):
        del obj.__dict__['multi_class']
    if hasattr(obj, 'estimators_'):
        for est in obj.estimators_:
            fix_multi_class(est)
    if hasattr(obj, 'steps'):
        for _, step in obj.steps:
            fix_multi_class(step)
    if hasattr(obj, 'estimators'):
        for _, est in obj.estimators:
            fix_multi_class(est)

# Load metadata
with open("model_metadata.json", "r") as f:
    metadata = json.load(f)

THRESHOLD = metadata.get("threshold", 0.5)

@app.route("/")
def home():
    return "VITALS Backend running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Load model safely
    model = load_model()

    # Extract features
    age = data.get("age")
    gender = data.get("gender")
    bmi = data.get("bmi")
    blood_pressure = data.get("blood_pressure")
    cholesterol = data.get("cholesterol")
    glucose = data.get("glucose")
    smoking_status = int(data.get("smoking_status", 0))
    physical_activity = data.get("physical_activity", 2)
    alcohol_intake = int(data.get("alcohol_intake", 0))

    # Derived features
    age_group = 0 if age <= 30 else 1 if age <= 45 else 2 if age <= 60 else 3
    bmi_category = 0 if bmi <= 18.5 else 1 if bmi <= 25 else 2 if bmi <= 30 else 3
    bp_category = 0 if blood_pressure <= 90 else 1 if blood_pressure <= 120 else 2 if blood_pressure <= 140 else 3
    glucose_category = 0 if glucose <= 70 else 1 if glucose <= 100 else 2 if glucose <= 126 else 3
    pulse_pressure = blood_pressure - (blood_pressure * 0.6)
    metabolic_risk = age * 0.25 + bmi * 0.20 + blood_pressure * 0.20 + glucose * 0.20 + cholesterol * 0.15
    bmi_glucose_interaction = bmi * glucose / 1000
    age_bp_interaction = age * blood_pressure / 1000

    smoking_bp_risk = smoking_status * blood_pressure / 140
    smoking_glucose_risk = smoking_status * glucose / 100
    activity_bmi_risk = (4 - physical_activity) * bmi / 30
    activity_glucose_risk = (4 - physical_activity) * glucose / 100
    alcohol_liver_risk = alcohol_intake * glucose / 100

    features = [
        age, gender, bmi, blood_pressure, cholesterol,
        glucose, smoking_status, physical_activity, alcohol_intake,
        age_group, bmi_category, bp_category, glucose_category,
        pulse_pressure, metabolic_risk, bmi_glucose_interaction, age_bp_interaction,
        smoking_bp_risk, smoking_glucose_risk, activity_bmi_risk, activity_glucose_risk, alcohol_liver_risk
    ]

    feature_names = [
        'age', 'gender', 'bmi', 'blood_pressure', 'cholesterol',
        'glucose', 'smoking_status', 'physical_activity', 'alcohol_intake',
        'age_group', 'bmi_category', 'bp_category', 'glucose_category',
        'pulse_pressure', 'metabolic_risk', 'bmi_glucose_interaction', 'age_bp_interaction',
        'smoking_bp_risk', 'smoking_glucose_risk', 'activity_bmi_risk', 'activity_glucose_risk', 'alcohol_liver_risk'
    ]

    features_df = pd.DataFrame([features], columns=feature_names)
    prob = model.predict_proba(features_df)[0][1]

    # Adjustments
    if alcohol_intake == 1:
        prob *= 1.15 if prob < 0.5 else 1.10
    elif alcohol_intake == 2:
        prob *= 1.25 if prob < 0.5 else 1.20

    if smoking_status == 1:
        prob *= 1.10
    elif smoking_status == 2:
        prob *= 1.25

    if physical_activity == 0 or physical_activity == "0":
        prob *= 1.15
    elif physical_activity == 1 or physical_activity == "1":
        prob *= 1.05
    elif physical_activity == 3 or physical_activity == "3":
        prob *= 0.92
    elif physical_activity == 4 or physical_activity == "4":
        prob *= 0.70

    prob = min(prob, 1.0)

    if prob < 0.15:
        risk_level = "Low Risk"
        classification = "Healthy"
        risk_color = "green"
    elif prob < 0.35:
        risk_level = "Moderate Risk"
        classification = "At Risk"
        risk_color = "orange"
    else:
        risk_level = "High Risk"
        classification = "High Risk"
        risk_color = "red"

    return (jsonify({
        "probability": float(prob),
        "risk_level": risk_level,
        "classification": classification,
        "risk_color": risk_color
    }), 200, {
        "Access-Control-Allow-Origin": "*"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)