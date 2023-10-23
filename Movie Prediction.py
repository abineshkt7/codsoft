# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer  # Import SimpleImputer

# Load the movie dataset (replace 'your_dataset.csv' with the actual dataset file)
df = pd.read_csv('your_dataset.csv', encoding='latin1')

# Data Preprocessing
# Handle missing values, duplicates, and data accuracy

# Drop columns with a lot of missing values (e.g., 'Actor 2', 'Actor 3')
df.drop(['Actor 2', 'Actor 3'], axis=1, inplace=True)

# Drop rows with missing values in critical columns (e.g., 'Rating', 'Votes')
df.dropna(subset=['Rating', 'Votes'], inplace=True)

# Impute missing values in other columns (e.g., 'Director', 'Genre')
df['Director'].fillna('Unknown Director', inplace=True)
df['Genre'].fillna('Unknown Genre', inplace=True)

# Data Cleaning: Handle special characters, brackets, and data type conversions

# Remove special characters from the 'Name' column (if needed)
df['Name'] = df['Name'].str.replace(r'^\W+', '', regex=True)

# Remove brackets from the 'Year' column (if needed)
df['Year'] = df['Year'].str.replace(r'[()]', '', regex=True)

# Extract numeric values from 'Duration' column and convert to integers (remove ' min')
df['Duration'] = df['Duration'].str.replace(r' min', '', regex=True)

# Handle missing values in 'Duration' by filling them with a default value (e.g., 0)
df['Duration'].fillna(0, inplace=True)

# Convert 'Duration' column to integers
df['Duration'] = df['Duration'].astype(int)

# Remove commas from 'Votes' column and convert to integers
df['Votes'] = df['Votes'].str.replace(',', '').astype(int)

# Feature Engineering: Encode categorical features (e.g., 'Genre', 'Director', 'Actor 1')

# Encode 'Genre' based on frequency
genre_counts = df['Genre'].value_counts()
df['Genre_encoded'] = df['Genre'].map(genre_counts)

# Encode 'Director' based on mean rating
director_mean_rating = df.groupby('Director')['Rating'].transform('mean')
df['Director_encoded'] = director_mean_rating

# Encode 'Actor 1' based on mean rating
actor_mean_rating = df.groupby('Actor 1')['Rating'].transform('mean')
df['Actor_encoded'] = actor_mean_rating

# Model Building
# Define features and target variable
X = df[['Year', 'Votes', 'Duration', 'Genre_encoded', 'Director_encoded', 'Actor_encoded']]
y = df['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in X using SimpleImputer (mean imputation)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using various metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation results
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

# You can now use this trained model to predict movie ratings based on input features.
