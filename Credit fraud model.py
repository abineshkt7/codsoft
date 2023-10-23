import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler  # Import RobustScaler
from sklearn.utils import resample

# Load the credit card fraud dataset
credit_card_data = pd.read_csv("creditcard.csv")

# Preprocess the data: Scale the 'Amount' column and normalize the 'Time' column
credit_card_data['Amount'] = RobustScaler().fit_transform(credit_card_data['Amount'].values.reshape(-1, 1))
credit_card_data['Time'] = (credit_card_data['Time'] - credit_card_data['Time'].min()) / (credit_card_data['Time'].max() - credit_card_data['Time'].min())

# Split the data into independent variables (features) and the target variable (Class)
X = credit_card_data.drop(columns='Class', axis=1)
y = credit_card_data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=4)

# Handle class imbalance by resampling the dataset
non_fraud = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Upsample the minority class (fraud) to balance the dataset
fraud_upsampled = resample(fraud, replace=True, n_samples=len(non_fraud), random_state=42)

# Combine the upsampled fraud data with the non-fraud data
balanced_data = pd.concat([non_fraud, fraud_upsampled])

# Split the balanced data into features and target
X_balanced = balanced_data.drop(columns='Class', axis=1)
y_balanced = balanced_data['Class']

# Train a logistic regression model on the balanced data
model = LogisticRegression()
model.fit(X_balanced, y_balanced)

# Make predictions using the trained model on the test set
predictions = model.predict(X_test)

# Evaluate the model's performance using precision, recall, and F1-score
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Display a full classification report
print(classification_report(y_test, predictions))
