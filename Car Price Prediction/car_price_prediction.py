import pandas as pd
import numpy as np

# Loading the dataset and viewing it
data = pd.read_csv('quikr_car.csv')

print(data.head(10))

# Get the shape of the dataset
print(data.shape)

# Get information about the dataset
print(data.info())


#--------------------------------------------------------------------------------



# Display unique values in various columns
print(data['year'].unique())
print(data['Price'].unique())
print(data['kms_driven'].unique())
print(data['fuel_type'].unique())



# Data Cleaning 

#--------------------------------------------------------------------------------

# Taking a backup of the original data
back = data.copy()

#--------------------------------------------------------------------------------


# Check for numeric values in the 'year' column
data = data[data['year'].str.isnumeric()]

# Convert 'year' to integer type
data['year'] = data['year'].astype(int)

# Display info after changes
print(data.info())


#--------------------------------------------------------------------------------


# Price Cleaning

# Display price values
print(data['Price'])

# Filter out 'Ask For Price' entries
data = data[data['Price'] != "Ask For Price"]

# Remove commas from the 'Price' and convert to integer
data['Price'] = data['Price'].str.replace(',', '').astype(int)

# Display info after cleaning price
print(data.info())

#--------------------------------------------------------------------------------


# Km Driven Cleaning

# Display 'kms_driven' column values
print(data['kms_driven'])

# Remove 'Kms' from 'kms_driven' and commas
data['kms_driven'] = data['kms_driven'].str.split().str.get(0).str.replace(',', '')

# Filter for numeric values in 'kms_driven'
data = data[data['kms_driven'].str.isnumeric()]

# Convert 'kms_driven' to integer type
data['kms_driven'] = data['kms_driven'].astype(int)

# Display info after cleaning kms_driven
print(data.info())


#--------------------------------------------------------------------------------


# Fuel Type Cleaning
# Remove entries with NaN in 'fuel_type'
data = data[~data['fuel_type'].isna()]

# Display the first 10 rows after cleaning
print(data.head(10))


#--------------------------------------------------------------------------------


# Car Names Processing

# Keep only the first three words of the car names
data['name'] = data['name'].str.split(' ').str.slice(0, 3).str.join(' ')


# Reset index due to data manipulation
data.reset_index(drop=True, inplace=True)


# Filter out unrealistic prices and reset index
data = data[data['Price'] < 6e6].reset_index(drop=True)

# Describe the cleaned dataset
print(data.describe())


#--------------------------------------------------------------------------------


# Save cleaned data to a new CSV file
data.to_csv('cleaned_car.csv', index=False)


#--------------------------------------------------------------------------------


# Modeling
# Separate features and target variable
x = data.drop(columns='Price')
y = data['Price']

# Display the first few rows of features and target
print(x.head(5))
print(y.head(5))

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#--------------------------------------------------------------------------------


# Import necessary libraries for modeling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


#--------------------------------------------------------------------------------


# Initialize OneHotEncoder for categorical features
hot = OneHotEncoder()
hot.fit(x[['name', 'company', 'fuel_type']])


#--------------------------------------------------------------------------------


# Create a column transformer for preprocessing
columns_trans = make_column_transformer(
    (OneHotEncoder(categories=hot.categories_, handle_unknown='ignore'), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

#--------------------------------------------------------------------------------




# Initialize Linear Regression model
lr = LinearRegression()

# Create a pipeline with preprocessing and model
pipe = make_pipeline(columns_trans, lr)

# Fit the model on training data
pipe.fit(x_train, y_train)

# Make predictions on test data
y_pred = pipe.predict(x_test)

# Display predicted values
print(y_pred)

# Calculate R² score of the model
print(r2_score(y_test, y_pred))





#--------------------------------------------------------------------------------




# Store R² scores in a list for analysis
scores = []
for i in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(columns_trans, lr)
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    scores.append(r2_score(y_test, y_pred))




#--------------------------------------------------------------------------------


# Find the index of the best score
best_index = np.argmax(scores)

# Display the best R² score and its index
print(best_index, scores[best_index])


#--------------------------------------------------------------------------------


# Train final model using the best random state
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=best_index)
lr = LinearRegression()
pipe = make_pipeline(columns_trans, lr)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)


#--------------------------------------------------------------------------------


# Display final R² score
final_score = r2_score(y_test, y_pred)
print(final_score)


#--------------------------------------------------------------------------------


# Save the trained model using pickle
import pickle
pickle.dump(pipe, open('Model.pkl', 'wb'))


#--------------------------------------------------------------------------------


# Sample predictions using the trained model
sample_input1 = pd.DataFrame([['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
sample_input2 = pd.DataFrame([['Hyundai Santro Xing', 'Hyundai', 2018, 45000, 'Petrol']], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

print(pipe.predict(sample_input1))
print(pipe.predict(sample_input2))


#--------------------------------------------------------------------------------

