import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

import gradio as gr

df = pd.read_csv("500_Person_Gender_Height_Weight_Index.csv")

df.columns = ["Gender", "Height_m", "Weight_kg", "Index"]

df.head()

df.tail()

df.info()

df.shape

df.isnull().sum()

bmi_labels = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity"
}

df = df[df["Index"].isin(bmi_labels)]
df["Category"] = df["Index"].map(bmi_labels)

sns.scatterplot(data=df, x="Height_m", y="Weight_kg", hue="Category")
plt.title("BMI Categories")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Category")
plt.title("Distribution of BMI Categories")
plt.xlabel("BMI Category")
plt.ylabel("Count")
plt.xticks(rotation=15)
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Category", y="Weight_kg")
plt.title("Weight Distribution per BMI Category")
plt.xlabel("BMI Category")
plt.ylabel("Weight (kg)")
plt.xticks(rotation=15)
plt.show()

corr = df[["Height_m", "Weight_kg", "Index"]].corr()
plt.figure(figsize=(5, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

df["Height_cm"] = df["Height_m"] * 100

X = df[["Height_cm", "Weight_kg"]]
le = LabelEncoder()
y = le.fit_transform(df["Category"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred, target_names=le.classes_))

import gradio as gr

def predict_bmi(height, weight):
    height_m = height / 100  # convert to meters
    bmi = weight / (height_m ** 2)
    if bmi <= 16:
        category = "Extremely Weak"
    elif bmi <= 18.5:
        category = "Weak"
    elif bmi <= 25:
        category = "Normal"
    elif bmi <= 30:
        category = "Overweight"
    else:
        category = "Obesity"
    return f"BMI: {bmi:.2f}, Category: {category}"

demo = gr.Interface(
    fn=predict_bmi,
    inputs=[gr.Number(label="Height (cm)"), gr.Number(label="Weight (kg)")],
    outputs="text",
    title="BMI Category Predictor"
)

demo.launch()
