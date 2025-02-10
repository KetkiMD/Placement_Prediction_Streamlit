
import streamlit as st
import pandas as pd
import joblib  

# Function to load the model and preprocessor based on course
def load_model(course):
    if course == "DAC":
        return joblib.load('latest_incremental_model_dac.pkl')
    elif course == "DBDA":
        return joblib.load('latest_incremental_model_dbda.pkl')
    else:
        return None

def load_preprocessor(course):
    if course == "DAC":
        return joblib.load('preprocessor_dac.pkl')
    elif course == "DBDA":
        return joblib.load('preprocessor_dbda.pkl')
    else:
        return None

# Function to predict placement probability
def predict_student_placement(student_data, course):
    model = load_model(course)
    preprocessor = load_preprocessor(course)
    
    if model is None or preprocessor is None:
        st.error("Error: Model or preprocessor not found for the selected course.")
        return None
    
    student_df = pd.DataFrame([student_data])
    student_transformed = preprocessor.transform(student_df)
    
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

# Streamlit UI
st.title("Student Placement Prediction")

# Select Course
course = st.selectbox("Select Course", ["DAC", "DBDA"])

# Input fields for student data based on the course
if course == "DAC":
    # DAC Course Inputs
    os_theory = st.number_input("Operating System Theory Marks", min_value=0, max_value=40, value=35)
    os_lab = st.number_input("Operating System Lab Marks", min_value=0, max_value=60, value=50)
    os_total = calculate_total(os_theory, os_lab)
    st.write(f"Operating System Total Marks: {os_total}")

    cpp_theory = st.number_input("C++ Theory Marks", min_value=0, max_value=40, value=30)
    cpp_lab = st.number_input("C++ Lab Marks", min_value=0, max_value=60, value=48)
    cpp_total = calculate_total(cpp_theory, cpp_lab)
    st.write(f"C++ Total Marks: {cpp_total}")

    java_theory = st.number_input("Java Theory Marks", min_value=0, max_value=40, value=32)
    java_lab = st.number_input("Java Lab Marks", min_value=0, max_value=60, value=50)
    java_total = calculate_total(java_theory, java_lab)
    st.write(f"Java Total Marks: {java_total}")

    dsa_theory = st.number_input("Data Structures Theory Marks", min_value=0, max_value=40, value=28)
    dsa_lab = st.number_input("Data Structures Lab Marks", min_value=0, max_value=60, value=45)
    dsa_total = calculate_total(dsa_theory, dsa_lab)
    st.write(f"Data Structures Total Marks: {dsa_total}")

    dbt_theory = st.number_input("Database Theory Marks", min_value=0, max_value=40, value=34)
    dbt_lab = st.number_input("Database Lab Marks", min_value=0, max_value=60, value=55)
    dbt_total = calculate_total(dbt_theory, dbt_lab)
    st.write(f"Database Total Marks: {dbt_total}")

    wpt_theory = st.number_input("Web Programming Theory Marks", min_value=0, max_value=40, value=33)
    wpt_lab = st.number_input("Web Programming Lab Marks", min_value=0, max_value=60, value=49)
    wpt_total = calculate_total(wpt_theory, wpt_lab)
    st.write(f"Web Programming Total Marks: {wpt_total}")

    wb_java_theory = st.number_input("Web-Based Java Theory Marks", min_value=0, max_value=40, value=29)
    wb_java_lab = st.number_input("Web-Based Java Lab Marks", min_value=0, max_value=60, value=47)
    wb_java_total = calculate_total(wb_java_theory, wb_java_lab)
    st.write(f"Web-Based Java Total Marks: {wb_java_total}")

    dotnet_theory = st.number_input("DotNet Theory Marks", min_value=0, max_value=40, value=35)
    dotnet_lab = st.number_input("DotNet Lab Marks", min_value=0, max_value=60, value=52)
    dotnet_total = calculate_total(dotnet_theory, dotnet_lab)
    st.write(f"DotNet Total Marks: {dotnet_total}")

    sdlc_theory = st.number_input("SDLC Theory Marks", min_value=0, max_value=40, value=34)
    sdlc_lab = st.number_input("SDLC Lab Marks", min_value=0, max_value=60, value=51)
    sdlc_total = calculate_total(sdlc_theory, sdlc_lab)
    st.write(f"SDLC Total Marks: {sdlc_total}")

    student_data = {
        "10th_percentage": st.number_input("10th Percentage", min_value=0.0, max_value=100.0, value=85.0),
        "grad_percentage": st.number_input("Graduation Percentage", min_value=0.0, max_value=100.0, value=78.5),
        "OS_Theory": os_theory,
        "OS_Lab": os_lab,
        "OS_Total": os_total,
        "OS_Status": st.selectbox("Operating System Status", ["Pass", "Fail"]),
        "CPP_Theory": cpp_theory,
        "CPP_Lab": cpp_lab,
        "CPP_Total": cpp_total,
        "CPP_Status": st.selectbox("C++ Status", ["Pass", "Fail"]),
        "Java_Theory": java_theory,
        "Java_Lab": java_lab,
        "Java_Total": java_total,
        "Java_Status": st.selectbox("Java Status", ["Pass", "Fail"]),
        "DSA_Theory": dsa_theory,
        "DSA_Lab": dsa_lab,
        "DSA_Total": dsa_total,
        "DSA_Status": st.selectbox("Data Structures Status", ["Pass", "Fail"]),
        "DBT_Theory": dbt_theory,
        "DBT_Lab": dbt_lab,
        "DBT_Total": dbt_total,
        "DBT_Status": st.selectbox("Database Status", ["Pass", "Fail"]),
        "WPT_Theory": wpt_theory,
        "WPT_Lab": wpt_lab,
        "WPT_Total": wpt_total,
        "WPT_Status": st.selectbox("Web Programming Status", ["Pass", "Fail"]),
        "WB_Java_Theory": wb_java_theory,
        "WB_Java_Lab": wb_java_lab,
        "WB_Java_Total": wb_java_total,
        "WB_Java_Status": st.selectbox("Web-Based Java Status", ["Pass", "Fail"]),
        "DotNet_Theory": dotnet_theory,
        "DotNet_Lab": dotnet_lab,
        "DotNet_Total": dotnet_total,
        "DotNet_Status": st.selectbox("DotNet Status", ["Pass", "Fail"]),
        "SDLC_Theory": sdlc_theory,
        "SDLC_Lab": sdlc_lab,
        "SDLC_Total": sdlc_total,
        "SDLC_Status": st.selectbox("SDLC Status", ["Pass", "Fail"]),
        "Total": st.number_input("Total Marks", min_value=0, max_value=800, value=150),
        "Total800": st.number_input("Total Marks out of 800", min_value=0, max_value=800, value=650),
        "Percentage": st.number_input("Percentage", min_value=0.0, max_value=100.0, value=85.5),
        "Grade": st.selectbox("Grade", ["A", "B", "C", "D", "E"]),
        "Result": st.selectbox("Result", ["Pass", "Fail"]),
        "Apti_EC_Grade": st.selectbox("Aptitude EC Grade", ["A", "B", "C", "D"]),
        "Project_Grade": st.selectbox("Project Grade", ["A", "B", "C", "D"]),
        "Higher_Edu_Percent": st.number_input("Higher Education Percentage", min_value=0.0, max_value=100.0, value=80.0),
        "Age": st.number_input("Age", min_value=0, max_value=100, value=21),
        "course": st.selectbox("Course", ["BTech", "MTech", "MBA", "BBA"])
    }
else:
    # DBDA Course Inputs
    dbms_theory = st.number_input("DBMS Theory Marks", min_value=0, max_value=40, value=35)
    dbms_lab = st.number_input("DBMS Lab Marks", min_value=0, max_value=60, value=50)
    dbms_total = calculate_total(dbms_theory, dbms_lab)
    st.write(f"DBMS Total Marks: {dbms_total}")

    java_theory = st.number_input("Java Theory Marks", min_value=0, max_value=40, value=30)
    java_lab = st.number_input("Java Lab Marks", min_value=0, max_value=60, value=48)
    java_total = calculate_total(java_theory, java_lab)
    st.write(f"Java Total Marks: {java_total}")

    python_r_theory = st.number_input("Python & R Theory Marks", min_value=0, max_value=40, value=32)
    python_r_lab = st.number_input("Python & R Lab Marks", min_value=0, max_value=60, value=50)
    python_r_total = calculate_total(python_r_theory, python_r_lab)
    st.write(f"Python & R Total Marks: {python_r_total}")

    stats_theory = st.number_input("Statistics Theory Marks", min_value=0, max_value=40, value=28)
    stats_lab = st.number_input("Statistics Lab Marks", min_value=0, max_value=60, value=45)
    stats_total = calculate_total(stats_theory, stats_lab)
    st.write(f"Statistics Total Marks: {stats_total}")

    data_viz_theory = st.number_input("Data Visualization Theory Marks", min_value=0, max_value=40, value=34)
    data_viz_lab = st.number_input("Data Visualization Lab Marks", min_value=0, max_value=60, value=55)
    data_viz_total = calculate_total(data_viz_theory, data_viz_lab)
    st.write(f"Data Visualization Total Marks: {data_viz_total}")

    big_data_theory = st.number_input("Big Data Theory Marks", min_value=0, max_value=40, value=33)
    big_data_lab = st.number_input("Big Data Lab Marks", min_value=0, max_value=60, value=49)
    big_data_total = calculate_total(big_data_theory, big_data_lab)
    st.write(f"Big Data Total Marks: {big_data_total}")

    linux_cloud_theory = st.number_input("Linux & Cloud Computing Theory Marks", min_value=0, max_value=40, value=29)
    linux_cloud_lab = st.number_input("Linux & Cloud Computing Lab Marks", min_value=0, max_value=60, value=47)
    linux_cloud_total = calculate_total(linux_cloud_theory, linux_cloud_lab)
    st.write(f"Linux & Cloud Computing Total Marks: {linux_cloud_total}")

    ml_theory = st.number_input("Machine Learning Theory Marks", min_value=0, max_value=40, value=35)
    ml_lab = st.number_input("Machine Learning Lab Marks", min_value=0, max_value=60, value=52)
    ml_total = calculate_total(ml_theory, ml_lab)
    st.write(f"Machine Learning Total Marks: {ml_total}")

    student_data = {
        "10th_percentage": st.number_input("10th Percentage", min_value=0.0, max_value=100.0, value=85.0),
        "grad_percentage": st.number_input("Graduation Percentage", min_value=0.0, max_value=100.0, value=78.5),
        "DBMSTheory": dbms_theory,
        "DBMSLab": dbms_lab,
        "DBMSTotal": dbms_total,
        "DBMSStatus": st.selectbox("DBMS Status", ["P", "F"]),
        "JavaTheory": java_theory,
        "JavaLab": java_lab,
        "JavaTotal": java_total,
        "JavaStatus": st.selectbox("Java Status", ["P", "F"]),
        "PythonRTheory": python_r_theory,
        "PythonRLab": python_r_lab,
        "PythonRTotal": python_r_total,
        "PythonRStatus": st.selectbox("Python & R Status", ["P", "F"]),
        "StatsTheory": stats_theory,
        "StatsLab": stats_lab,
        "StatsTotal": stats_total,
        "StatsStatus": st.selectbox("Statistics Status", ["P", "F"]),
        "DataVizTheory": data_viz_theory,
        "DataVizLab": data_viz_lab,
        "DataVizTotal": data_viz_total,
        "DataVizStatus": st.selectbox("Data Visualization Status", ["P", "F"]),
        "BigDataTheory": big_data_theory,
        "BigDataLab": big_data_lab,
        "BigDataTotal": big_data_total,
        "BigDataStatus": st.selectbox("Big Data Status", ["P", "F"]),
        "LinuxCloudTheory": linux_cloud_theory,
        "LinuxCloudLab": linux_cloud_lab,
        "LinuxCloudTotal": linux_cloud_total,
        "LinuxCloudStatus": st.selectbox("Linux & Cloud Computing Status", ["P", "F"]),
        "MLTheory": ml_theory,
        "MLLab": ml_lab,
        "MLTotal": ml_total,
        "MLStatus": st.selectbox("Machine Learning Status", ["P", "F"]),
        "Total": st.number_input("Total Marks", min_value=0, max_value=800, value=650),
        "Percentage": st.number_input("Percentage", min_value=0.0, max_value=100.0, value=85.5),
        "Result": st.selectbox("Result", ["Pass", "Fail"]),
        "Higher_Edu_Percent": st.number_input("Higher Education Percentage", min_value=0.0, max_value=100.0, value=80.0)
    }

# Button to predict placement probability
if st.button("Predict Placement Probability"):
    prediction_result = predict_student_placement(student_data, course)
    if prediction_result:
        st.success(f"Placement Prediction: {prediction_result['Placement Prediction']}")
        st.write(f"Probability of Placement (Yes): {prediction_result['Probability Yes']}%")
        st.write(f"Probability of Placement (No): {prediction_result['Probability No']}%")