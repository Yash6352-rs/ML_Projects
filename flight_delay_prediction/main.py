import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib

df = pd.read_csv("DelayedFlights.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

# using only 10000 rows so pc runs properly
df = df.sample(n=10000, random_state=42)

# remove nan row
df = df.dropna(subset=['DepDelay', 'CRSDepTime', 'Distance', 'UniqueCarrier', 'Origin', 'Dest'])

df['Delayed'] = (df['DepDelay'] > 15).astype(int)

# Delay Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Delayed', palette='Set2')
plt.title("Flight Delay(0 = On-Time, 1 = Delayed)")
plt.tight_layout()
plt.show()

# Delays by Day of Week
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='DayOfWeek', hue='Delayed', palette='Set2')
plt.title("Delays by Day of the Week")
plt.tight_layout()
plt.show()

# Distance by Delay
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='Delayed', y='Distance', palette='Set3')
plt.title("Flight Distance by Delay Status")
plt.tight_layout()
plt.show()

# Delay Rate by Airline
plt.figure(figsize=(12, 5))
carrier_delay = df.groupby('UniqueCarrier')['Delayed'].mean().sort_values()
carrier_delay.plot(kind='bar', color='skyblue')
plt.title("Average Delay Rate by Airline")
plt.ylabel("Proportion of Delays")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
corr = df[['DepDelay', 'CRSDepTime', 'Distance', 'DayOfWeek', 'Delayed']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Feature selection
features = ['UniqueCarrier', 'CRSDepTime', 'DayOfWeek', 'Distance', 'Origin', 'Dest']
target = 'Delayed'

X = df[features]
y = df[target]

# text to number convert
label_encoders = {}
for col in ['UniqueCarrier', 'Origin', 'Dest']:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=25, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, 'flight_delay_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

