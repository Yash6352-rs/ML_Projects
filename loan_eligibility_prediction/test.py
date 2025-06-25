import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

test_df = pd.read_csv("loan-test.csv")
test_df.drop(columns=["Loan_ID"], inplace=True)

# Fill missing values
test_df['Gender'].fillna(test_df['Gender'].mode()[0], inplace=True)
test_df['Married'].fillna(test_df['Married'].mode()[0], inplace=True)
test_df['Dependents'].fillna(test_df['Dependents'].mode()[0], inplace=True)
test_df['Self_Employed'].fillna(test_df['Self_Employed'].mode()[0], inplace=True)
test_df['Credit_History'].fillna(test_df['Credit_History'].mode()[0], inplace=True)
test_df['Loan_Amount_Term'].fillna(test_df['Loan_Amount_Term'].mode()[0], inplace=True)
test_df['LoanAmount'].fillna(test_df['LoanAmount'].mode()[0], inplace=True)

le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']:
    test_df[col] = le.fit_transform(test_df[col])

features = ['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Dependents']

test_df = test_df[features]

model = joblib.load('loan_model.pkl')

predictions = model.predict(test_df)
print("Loan Approval Predictions (1 = Approved, 0 = Not Approved):")
print(predictions)
