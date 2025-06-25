import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load('loan_model.pkl')

st.title("Loan Eligibility Predictor")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=1)
Loan_Amount_Term = st.number_input("Loan Term (in days)", min_value=1)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict"):
    input = pd.DataFrame({
        "Gender": [Gender],
        "Married": [Married],
        "Education": [Education],
        "Self_Employed": [Self_Employed],
        "ApplicantIncome": [ApplicantIncome],
        "CoapplicantIncome": [CoapplicantIncome],
        "LoanAmount": [LoanAmount],
        "Loan_Amount_Term": [Loan_Amount_Term],
        "Credit_History": [Credit_History],
        "Property_Area": [Property_Area],
        "Dependents": [Dependents]
    })

    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']:
        le = LabelEncoder()
        le.fit(['Male', 'Female'] if col == 'Gender' else
               ['Yes', 'No'] if col in ['Married', 'Self_Employed'] else
               ['Graduate', 'Not Graduate'] if col == 'Education' else
               ['Urban', 'Semiurban', 'Rural'] if col == 'Property_Area' else
               ['0', '1', '2', '3+'])
        input[col] = le.transform(input[col])

    input = input[['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area','Dependents']]

    prediction = model.predict(input)[0]

    if prediction == 1:
        st.success("Eligible for Loan")
    else:
        st.error("Not Eligible for Loan")
