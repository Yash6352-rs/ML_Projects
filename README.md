# ML_Projects

This repository showcases beginner to intermediate machine learning projects developed during my learning journey. Each project demonstrates end-to-end ML workflows including data preprocessing, model training, evaluation, and deployment using web frameworks like Streamlit and Gradio.

---

## Projects Overview

### 1. **BMI Category Prediction**  
This project calculates an individual's Body Mass Index (BMI) using height and weight inputs and classifies them into five categories:  
`Extremely Weak`, `Weak`, `Normal`, `Overweight`, or `Obesity`.

- **Dataset:** [Real-world height-weight data ](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex) 
- **Model:** Decision Tree Classifier  
- **Steps Covered:**  
  - BMI calculation formula  
  - BMI category labeling  
  - Model training and prediction  
  - Web deployment with Gradio   

---

### 2. **Flight Delay Prediction** *(Streamlit App)*  
This project predicts whether a flight will be delayed by more than 15 minutes using key features such as the airline, day of the week, scheduled departure time, and distance.

- **Dataset:** [Cleaned flight delay dataset](https://www.kaggle.com/datasets/giovamata/airlinedelaycauses) 
- **Model:** Random Forest Classifier  
- **Steps Covered:**  
  - Exploratory Data Analysis (EDA)  
  - Feature encoding (Label Encoding)  
  - Model training and testing  
  - Evaluation with accuracy and classification report  
  - Deployment using Streamlit   
- **Interface:** Intuitive Streamlit form for input and delay status output

---

### 3. **Loan Eligibility Prediction** *(Streamlit App)*  
This ML project determines whether a loan application is likely to be approved based on applicant information such as income, marital status, education, loan amount, credit history, and more.

- **Dataset:** [Loan Eligible Dataset](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset)  
- **Model:** Logistic Regression  
- **Steps Covered:**  
  - Handling missing data  
  - Data visualization and correlation heatmaps  
  - Feature encoding with LabelEncoder  
  - Model training and evaluation  
  - Deployment via Streamlit app  
- **Interface:** User-friendly Streamlit interface to enter applicant info and get instant eligibility results

---

## How to Run Locally
You can run the **Flight Delay Prediction** and **Loan Eligibility Prediction** projects locally using Streamlit.

### 1. Clone the Repository
git clone https://github.com/Yash6352-rs/ML_Projects.git

cd ML_Projects

### 2. pip install pandas numpy scikit-learn streamlit seaborn matplotlib

### 3. Run the Apps

  ## BMI Category Prediction (Gradio)
      cd bmi_category_predictor
      python app.py

  ## Flight Delay Prediction
      cd flight_delay_prediction
      streamlit run main.py
      
  ## Loan Eligibility Prediction
      cd loan_eligibility_prediction
      streamlit run app.py
