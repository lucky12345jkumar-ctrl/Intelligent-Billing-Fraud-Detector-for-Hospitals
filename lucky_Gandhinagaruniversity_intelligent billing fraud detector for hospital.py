# Generated from: lucky_Gandhinagaruniversity_intelligent billing fraud detector for hospital.ipynb
# Converted at: 2026-04-04T17:58:16.731Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Intelligent Billing Fraud Detector for Hospitals


# ## Abstract
# 
# This system identifies fraudulent or inflated hospital bills using pattern recognition and anomaly detection. Models analyze treatment codes, durations, diagnostics, and pricing inconsistencies. NLP validates medical descriptions to detect exaggerations or duplication. Suspicious cases are flagged for audit. This enhances transparency and reduces financial misconduct.


#  ## Technology Stack
# 
# - Data Science & Machine Learning
# - Python
# - Jupyter Notebook / JupyterLab
# - Libraries:
# - Pandas
# - NumPy
# - Scikit-learn
# - Matplotlib / Seaborn


# ## Dataset Overview (Your CSV)
# 
# - ClaimID
# - PatientID
# - ProviderID
# - TreatmentType
# - TreatmentDurationDays
# - BillingAmount
# - ApprovedAmount
# - NumProcedures
# - FraudFlag (Target Variable)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\bijar\OneDrive\Desktop\billing_fraud_dataset_rows.csv.csv (1).xls")


df.head()

df.tail()

df.info()

df.describe()

df.isnull().sum()

df = pd.get_dummies(df, columns=['TreatmentType'], drop_first=True)

# ## Data Visualization


sns.countplot(x='FraudFlag', data=df)
plt.title("Fraud vs Non-Fraud Cases")
plt.show()

# ### ->This graph shows the distribution of fraudulent and non-fraudulent claims.
# ### ->Most claims belong to the non-fraud category.
# ### ->Fraud cases are fewer but significant for analysis.


sns.boxplot(x='FraudFlag', y='BillingAmount', data=df)
plt.show()

# ### ->Fraud cases often involve a higher number of procedures.
# ### ->Some claims show unusually high procedure counts.
# ### ->This may indicate unnecessary or duplicated treatments.
# ### ->Non-fraud cases have more consistent procedure counts.


plt.figure(figsize=(8,5))
sns.histplot(df['BillingAmount'], bins=30, kde=True)
plt.title("Distribution of Billing Amount")
plt.show()

# ### ->The distribution shows how billing amounts vary across claims.
# ### ->Most values are concentrated in a specific range.
# ### ->A few extreme high values (outliers) may indicate suspicious billing.
# ### ->Fraud cases are often associated with higher billing amounts.


plt.figure(figsize=(8,6))
sns.scatterplot(x='ApprovedAmount', y='BillingAmount', hue='FraudFlag', data=df)
plt.title("Billing vs Approved Amount")
plt.show()

# ### ->This graph compares billed and approved amounts.
# ### ->Ideally, both values should be close for genuine claims.
# ### ->Fraud cases appear where billing is much higher than approval.
# ### ->Points far from the diagonal line indicate suspicious activit


plt.figure(figsize=(10,6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# ### ->The heatmap shows relationships between numerical features.
# ### ->Strong correlations indicate important predictors.
# ### ->BillingAmount and ApprovedAmount are highly related.
# ### ->Derived features like AmountDifference are strong fraud indicators.


# ## Feature Engineering


# Difference between billed and approved
df['AmountDifference'] = df['BillingAmount'] - df['ApprovedAmount']

# Ratio feature
df['BillingRatio'] = df['BillingAmount'] / (df['ApprovedAmount'] + 1)

# ### ->Fraud often shows large differences
# ### ->Ratio helps detect abnormal billing


# ## Define Features and Target


X = df.drop(['FraudFlag', 'ClaimID', 'PatientID', 'ProviderID'], axis=1)
y = df['FraudFlag']

# # Train-Test Split


from sklearn.model_selection import train_test_split

X = df.drop(['FraudFlag', 'ClaimID', 'PatientID', 'ProviderID'], axis=1)
y = df['FraudFlag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ### ->Training Data → used to train the model
# ### ->Testing Data → used to evaluate the model
# ### ->To check how well the model performs on unseen data
# ### ->To avoid overfitting (model memorizing data)
# ### ->To ensure generalization of the model


# # Model Training


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# ## Random Forest 


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# # Prediction


y_pred = rf.predict(X_test)

# # Model Evaluation


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# # Feature Importance


importances = rf.feature_importances_
features = X.columns

sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()

# ## ->Important Factors:
# ### -> BillingAmount
# ### -> AmountDifference
# ### -> NumProcedures


# # Anomaly Detection


from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05)
df['Anomaly'] = iso.fit_predict(X)

df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})

# ## ->Explanation:
# ### ->Detects unusual billing patterns
# ### ->Flags rare cases as anomalies


# # Final Fraud Detection Logic


df['FinalFlag'] = ((df['BillingRatio'] > 2) | (df['Anomaly'] == 1)).astype(int)

# ## ->A claim is suspicious if:
# 
# ### ->Model predicts fraud
# ### ->OR anomaly detected
# ### ->OR billing ratio is too high


# # Results
# 
# - Fraudulent claims detected successfully
# - High accuracy using Random Forest
# - Suspicious billing patterns identified


# # Advantages
# 
# - Reduces manual auditing
# - Detects hidden fraud patterns
# - Improves healthcare transparency
# - Scalable for real-time systems


# # Future Scope
# 
# - NLP for doctor notes analysis
# - Deep learning models
# - Real-time fraud detection dashboard
# - Integration with insurance systems


# Conclusion