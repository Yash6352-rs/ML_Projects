# ML_Projects

This repository showcases beginner to intermediate machine learning projects developed during my learning journey. Each project demonstrates end-to-end ML workflows including data preprocessing, model training, evaluation, and deployment using web frameworks like Streamlit and Gradio.

These projects are useful for learning fundamental ML concepts, working with real-world datasets, and building interactive web applications for model inference.

This repository aims to serve as a **learning resource**, **personal showcase**, and **foundation** for future ML apps.

---

## Projects Overview

### 1. **BMI Category Prediction**  *(Gradio)*

This project calculates an individual's Body Mass Index (BMI) using height and weight inputs and classifies them into five categories:  
`Extremely Weak`, `Weak`, `Normal`, `Overweight`, or `Obesity`.

- **Dataset:** [500 Person Gender-Height-Weight-Body Mass Index ](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex) 
- **Model:** Decision Tree Classifier  
- **Key Steps:**  
  - BMI calculation formula and category labeling  
  - Model training with BMI classification  
  - Export model using `joblib`  
  - Deployment using **Gradio UI**  
- **Development Platform:** Originally developed on **Google Colab** 

---

### 2. **Flight Delay Prediction** *(Streamlit App)*  

This project predicts whether a flight will be delayed by more than 15 minutes using key features such as the airline, day of the week, scheduled departure time, and distance.

- **Dataset:** [Airlines Delay](https://www.kaggle.com/datasets/giovamata/airlinedelaycauses) 
- **Model:** Random Forest Classifier  
- **Key Steps:**  
  - Exploratory Data Analysis (EDA)  
  - Label Encoding of categorical variables  
  - Model training and accuracy evaluation  
  - Deployment using **Streamlit**  
- **Interface:** Streamlit form to enter flight details and display delay prediction  
---

### 3. **Loan Eligibility Prediction** *(Streamlit App)*  

This ML project determines whether a loan application is likely to be approved based on applicant information such as income, marital status, education, loan amount, credit history, and more.

- **Dataset:** [Loan Eligible Dataset](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset)  
- **Model:** Logistic Regression  
- **Key Steps:**  
  - Handling missing values  
  - Label encoding of categorical features  
  - Data visualization (histograms, boxplots, heatmaps)  
  - Model training and evaluation using metrics  
  - Deployment using **Streamlit**  
- **Interface:** Interactive Streamlit app that accepts user input and returns loan eligibility status  

---

## 📁 Project Structure
│
├── bmi_category_predictor/ → Gradio App + EDA + model
├── flight_delay_prediction/ → Streamlit App + EDA + trained model
├── loan_eligibility_prediction/ → Streamlit App + EDA + model
└── README.md → Project overview

## How to Run Locally
You can run the **Flight Delay Prediction** and **Loan Eligibility Prediction** projects locally using Streamlit.

### 1. Clone the Repository
git clone https://github.com/Yash6352-rs/ML_Projects.git

cd ML_Projects

### 2. Install Dependencies
pip install pandas numpy scikit-learn streamlit seaborn matplotlib

### 3. Running the Projects

  ## BMI Category Prediction (Gradio)
      cd bmi_category_predictor
      python app.py

  ## Flight Delay Prediction (Streamlit App)
      cd flight_delay_prediction
      streamlit run main.py
      
  ## Loan Eligibility Prediction (Streamlit App)
      cd loan_eligibility_prediction
      streamlit run app.py

### 📈 Model Evaluation

Each project prints performance metrics such as:
  -> Accuracy
  -> Confusion Matrix
  -> Classification Report

In addition, the Loan Eligibility app provides:
  -> Visual Insights (count plots, boxplots, correlation heatmaps)
  -> EDA-driven feature understanding

### 🤝 Contributing

Pull requests are welcome! If you want to improve the models, refactor the code, or build new apps, feel free to fork and contribute.
For major changes, open an issue first to discuss what you would like to change.


### 📬 Contact
Created with ❤️ by Yash Panchal
For questions, ideas, or collaborations, reach me via GitHub.



