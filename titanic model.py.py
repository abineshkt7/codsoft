import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('tested.csv')

def data_overview(df, message):
    print(f'{message}:\n')
    print("Rows:", df.shape[0])
    print("\nNumber of features:", df.shape[1])
    print("\nFeatures:")
    print(df.columns.tolist())
    print("\nMissing values:", df.isnull().sum().values.sum())
    print("\nUnique values:")
    print(df.nunique())

data_overview(df, 'Overview of the dataset')

# Data preprocessing
df.drop(columns=['PassengerId', 'Name', 'Embarked', 'Cabin'], inplace=True)

# Fill missing values in Age and Fare columns
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Visualizations
plt.figure(figsize=(14,7))
plt.subplot(2,2,1)
sns.boxplot(x='Sex', y='Age', data=df)

plt.subplot(2,2,2)
sns.histplot(df['Fare'], color='g')

plt.subplot(2,2,3)
sns.histplot(df['Age'], color='g')

plt.subplot(2,2,4)
sns.countplot(x='Sex', data=df)

plt.tight_layout()
plt.show()

# Label Encoding
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])

# Extract numeric part from 'Ticket' column using regular expressions
df['Ticket'] = df['Ticket'].apply(lambda x: re.sub(r'\D', '', x))

# Split the data into features (X) and target (y)
X = df.drop(['Survived'], axis=1)
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# List of models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(max_iter=1000)  # You can adjust max_iter as needed
}

# Train and evaluate each model
model_results = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, Y_train)
    Y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)
    model_results[model_name] = {'Accuracy': accuracy, 'Classification Report': report}

# Visualize results in a pie chart
accuracy_values = [result['Accuracy'] for result in model_results.values()]
model_names = model_results.keys()

plt.figure(figsize=(8, 8))
plt.pie(accuracy_values, labels=model_names, autopct='%1.1f%%', startangle=140)
plt.title('Model Accuracy')
plt.show()

# Display model results
for model_name, result in model_results.items():
    print(f"{model_name} - Accuracy: {result['Accuracy']:.2f}")
    print(f"Classification Report:\n{result['Classification Report']}\n")
