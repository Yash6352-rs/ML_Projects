import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

train_df = pd.read_csv("loan-train.csv")
train_df.drop(columns=['Loan_ID'], inplace=True)

print(train_df.info())
print(train_df.isnull().sum())
print(train_df.shape)
print(train_df.head())

# Fill missing values
train_df['Gender'].fillna(train_df['Gender'].mode()[0], inplace=True)
train_df['Married'].fillna(train_df['Married'].mode()[0], inplace=True)
train_df['Dependents'].fillna(train_df['Dependents'].mode()[0], inplace=True)
train_df['Self_Employed'].fillna(train_df['Self_Employed'].mode()[0], inplace=True)
train_df['Credit_History'].fillna(train_df['Credit_History'].mode()[0], inplace=True)
train_df['Loan_Amount_Term'].fillna(train_df['Loan_Amount_Term'].mode()[0], inplace=True)
train_df['LoanAmount'].fillna(train_df['LoanAmount'].mode()[0], inplace=True)

# approved or not approved
plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='Loan_Status')
plt.title("Loan Status Approved or Not Approved")
plt.show()

# differnt categorical columns vs Loan_Status
plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='Gender', hue='Loan_Status')
plt.title("Loan Status according to Gender")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='Married', hue='Loan_Status')
plt.title("Loan Status according to Married or Not Married")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='Education', hue='Loan_Status')
plt.title("Loan Status according to Education")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='Property_Area', hue='Loan_Status')
plt.title("Loan Status according to Property Area")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='Dependents', hue='Loan_Status')
plt.title("Loan Status according to Dependents")
plt.show()

# Applicant Income Distribution
plt.figure(figsize=(6, 4))
sns.histplot(train_df['ApplicantIncome'])
plt.title("Applicant Income Distribution")
plt.show()

# Amount and Loan_Status
plt.figure(figsize=(6, 4))
sns.boxplot(data=train_df, x='Loan_Status', y='LoanAmount')
plt.title("Amount and Loan Status")
plt.show()

# LoanAmount and Credit_History
plt.figure(figsize=(6, 4))
sns.boxplot(data=train_df, x='Credit_History', y='LoanAmount', hue='Loan_Status')
plt.title("LoanAmount vs Credit History")
plt.show()

# Correlation heatmap
numeric_df = train_df.select_dtypes(include=['number']) 
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents', 'Loan_Status']:
      train_df[col] = le.fit_transform(train_df[col])


features = ['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Dependents']

X = train_df[features]
Y = train_df['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

dump(model, "loan_model.pkl")
