import streamlit as st
import pandas as pd
import joblib
import io
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import boto3
import json
import os


theory_cols = ['DBMSTheory', 'JavaTheory', 'PythonRTheory', 'StatsTheory', 
               'DataVizTheory', 'BigDataTheory', 'LinuxCloudTheory', 'MLTheory']
lab_cols = ['DBMSLab', 'JavaLab', 'PythonRLab', 'StatsLab', 'DataVizLab', 
            'BigDataLab', 'LinuxCloudLab', 'MLLab']
percent_cols = ['10th_percentage', 'grad_percentage', 'Higher_Edu_Percent']
one_hot_col = ['branch_cleaned']
num_cols = ['10th_percentage', 'grad_percentage', 'DBMSTheory', 'DBMSLab', 'JavaTheory',
            'JavaLab', 'PythonRTheory', 'PythonRLab', 'StatsTheory', 'StatsLab', 
            'DataVizTheory', 'DataVizLab', 'BigDataTheory', 'BigDataLab', 
            'LinuxCloudTheory', 'LinuxCloudLab', 'MLTheory', 'MLLab', 
             'Higher_Edu_Percent']

cat_cols = ['Grade',  'aptigrade', 'projectgrade', 'branch_cleaned']



def preprocessing(X_df,preprocessor):
    X_transformed = preprocessor.transform(X_df)
    all_col_names = num_cols + cat_cols

    X_df = pd.DataFrame(X_transformed, columns=all_col_names)

    degree_mapping = {
        "Computer": 5,
        "Electronics and Telecommunication": 4,
        "Electrical": 3,
        "Mechanical": 3,
        "Civil": 2,
        "BE": 2,
        "BSc": 2,
        "Mathematics": 2,
        "Chemical": 1,
        "Instrumentation": 1,
        "Physics": 1
    }

    # Apply encoding
    X_df["branch_cleaned"] = X_df["branch_cleaned"].map(degree_mapping).fillna(0)


    X_df[theory_cols] = X_df[theory_cols] / 40
    X_df[lab_cols] = X_df[lab_cols] / 60
    X_df[percent_cols] = X_df[percent_cols] / 100

    grade_cols = ['Grade', 'aptigrade', 'projectgrade']
    grade_mapping = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0, 'Fail': 0, 'Pass': 1}

    for col in grade_cols:
        if col in X_df.columns:
            X_df[col] = X_df[col].map(grade_mapping).fillna(0)


    return X_df


import pickle

course="DBDA"

def load_model():
    model_path = "dbda_latest_model.pkl"
    with open(model_path, "rb") as file:
        model = joblib.load(file)
    return model  # Ensure it returns the loaded model

def load_preprocessor():
    preprocessor_path = "dbda_latest_processor.pkl"
    with open(preprocessor_path, "rb") as file:
        preprocessor = joblib.load(file)
    return preprocessor  # Ensure it returns the loaded preprocessor


# Function to predict placement probability
def predict_student_placement(student_data):
    model = load_model()
    preprocessor = load_preprocessor()
    
    if model is None or preprocessor is None:
        return None
    
    student_df = pd.DataFrame([student_data])
    student_transformed = preprocessing(student_df,preprocessor)
    
    prediction = model.predict(student_transformed)
    probabilities = model.predict_proba(student_transformed)
    
    prob_yes = probabilities[0][list(model.classes_).index("Yes")]
    prob_no = probabilities[0][list(model.classes_).index("No")]

    return {
        "Placement Prediction": "Yes" if prediction[0] == "Yes" else "No",
        "Probability Yes": round(prob_yes * 100, 2),
        "Probability No": round(prob_no * 100, 2)
    }

# Function to calculate total marks
def calculate_total(theory, lab):
    return theory + lab






