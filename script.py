# script.py is a file purposedly created to demonstrate in the repository overview that Python is a major component of the project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Simulated analysis (can be anything that matches your notebook)
df = pd.read_csv("Churn_Modelling.csv")
print(df.head())

# RandomForestClassifier
X = df[['CreditScore', 'Age', 'Balance']]  # Example features
y = df['Exited']
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

print(f"Model Accuracy: {clf.score(X, y)}")
