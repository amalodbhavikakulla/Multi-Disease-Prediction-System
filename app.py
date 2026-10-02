from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
diabetes_model = joblib.load("diabetes_random_forest.pkl")
heart_model = joblib.load("heart_disease_model.pkl")
liver_model = joblib.load("liver_disease_model.pkl")


# ==========================================
# LOAD MODELS
# ==========================================

# Diabetes model
diabetes_model = joblib.load("diabetes_random_forest.pkl")

# Heart Disease model
heart_model = joblib.load("heart_disease_model.pkl")
# Load Liver Disease model
liver_model = joblib.load("liver_disease_model.pkl")

# Load Liver Disease feature columns
liver_features = joblib.load("liver_feature_columns.pkl")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# DIABETES PAGE
# ==========================================

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


# ==========================================
# DIABETES PREDICTION
# ==========================================

@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():

    pregnancies = float(request.form["Pregnancies"])
    glucose = float(request.form["Glucose"])
    blood_pressure = float(request.form["BloodPressure"])
    skin_thickness = float(request.form["SkinThickness"])
    insulin = float(request.form["Insulin"])
    bmi = float(request.form["BMI"])
    diabetes_pedigree = float(request.form["DiabetesPedigreeFunction"])
    age = float(request.form["Age"])

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    prediction = diabetes_model.predict(input_data)[0]

    if prediction == 1:
        result = "Higher Risk of Diabetes"
    else:
        result = "Lower Risk of Diabetes"

    return render_template(
        "diabetes.html",
        prediction=result
    )


# ==========================================
# HEART DISEASE PAGE
# ==========================================

@app.route("/heart")
def heart():
    return render_template("heart.html")


# ==========================================
# HEART DISEASE PREDICTION
# ==========================================

@app.route("/predict_heart", methods=["POST"])
def predict_heart():

    age = float(request.form["age"])
    sex = float(request.form["sex"])
    cp = float(request.form["cp"])
    trestbps = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    fbs = float(request.form["fbs"])
    restecg = float(request.form["restecg"])
    thalach = float(request.form["thalach"])
    exang = float(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = float(request.form["slope"])
    ca = float(request.form["ca"])
    thal = float(request.form["thal"])

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = heart_model.predict(input_data)[0]

    if prediction == 1:
        result = "Higher Risk of Heart Disease"
    else:
        result = "Lower Risk of Heart Disease"

    return render_template(
        "heart.html",
        prediction=result
    )


# ==========================================
# LIVER PAGE
# ==========================================

@app.route("/liver")
def liver():
    return render_template("liver.html")


@app.route("/predict_liver", methods=["POST"])
def predict_liver():

    # Get values from the form
    age = float(request.form["age"])
    gender = request.form["gender"]
    tot_bilirubin = float(request.form["tot_bilirubin"])
    direct_bilirubin = float(request.form["direct_bilirubin"])
    tot_proteins = float(request.form["tot_proteins"])
    albumin = float(request.form["albumin"])
    ag_ratio = float(request.form["ag_ratio"])
    sgpt = float(request.form["sgpt"])
    sgot = float(request.form["sgot"])
    alkphos = float(request.form["alkphos"])

    # Convert Gender into the same format used during training
    gender_male = 1 if gender == "Male" else 0

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "age": age,
        "tot_bilirubin": tot_bilirubin,
        "direct_bilirubin": direct_bilirubin,
        "tot_proteins": tot_proteins,
        "albumin": albumin,
        "ag_ratio": ag_ratio,
        "sgpt": sgpt,
        "sgot": sgot,
        "alkphos": alkphos,
        "gender_Male": gender_male
    }])

    # Make prediction
    prediction = liver_model.predict(input_data)[0]

    # Convert prediction into message
    if prediction == 1:
        result = "Higher Risk of Liver Disease"
    else:
        result = "Lower Risk of Liver Disease"

    return render_template(
        "liver.html",
        prediction=result
    )
# ==========================================
# RUN APPLICATION
# ==========================================
@app.route("/eda")
def eda():
    return render_template("eda.html")


@app.route("/eda/diabetes")
def diabetes_eda():
    return render_template("diabetes_eda.html")


@app.route("/eda/heart")
def heart_eda():
    return render_template("heart_eda.html")


@app.route("/eda/liver")
def liver_eda():
    return render_template("liver_eda.html")


if __name__ == "__main__":
    app.run(debug=True)