import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('data/linear_dataset.csv')

# Explore the dataset
print(data.head())

# Data preprocessing
# Convert categorical variables into numerical using one-hot encoding
# Drop 'Car_Name' column as it won't contribute to the prediction
data = pd.get_dummies(data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

# Prepare the data
X = data.drop(['Selling_Price', 'Car_Name'], axis=1)
y = data['Selling_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing
numeric_features = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Append regression model to pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualize the results
fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Selling Price', 'y': 'Predicted Selling Price'}, 
                 title='Actual vs Predicted Selling Price')
fig.show()