# Car Price Prediction 🚗💰

This project is a car price prediction model that uses a machine learning approach to estimate the price of used cars based on various features. The project includes a data cleaning and modeling script, and a Flask-based web application for user-friendly predictions.

### **Project Overview**

The core of this project is a machine learning model built using **Scikit-learn's `LinearRegression`** and a **`OneHotEncoder`** for handling categorical data. The model is trained on a dataset of used cars and their attributes, such as year, kilometers driven, fuel type, and company.

The project consists of two main files:

  * `carprediction.py`: The Python script for data preprocessing, model training, and saving the model using `pickle`.
  * `app.py`: A **Flask** web application that loads the trained model and serves a web interface for making predictions.

-----

### **Key Features**

  * **Data Cleaning:** The script effectively handles inconsistencies in the raw dataset, including:
      * Converting `year`, `price`, and `kms_driven` to numerical formats.
      * Removing non-numeric and irrelevant entries.
      * Standardizing car names to improve data quality.
  * **Machine Learning Model:** A **Linear Regression** model is used for its simplicity and interpretability. The model's performance is optimized by running multiple training iterations to find the best random state, ensuring a robust and reliable prediction.
  * **Web Application:** A user-friendly web interface, built with **Flask**, allows users to input car details and receive an estimated price in real-time.
  * **Modularity:** The project separates the model training and serving components, making it easy to update the model without changing the application logic.

-----

### **Technologies Used**

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python | Core project language. |
| **Data Manipulation** | Pandas, NumPy | Data cleaning and numerical operations. |
| **Machine Learning** | Scikit-learn | Model training, preprocessing, and evaluation. |
| **Web Framework** | Flask | Building the web application. |
| **Model Serialization** | Pickle | Saving and loading the trained model. |
| **Web Technologies** | HTML, CSS | Creating the front-end interface. |
| **Data Source** | `quikr_car.csv` | Dataset for model training. |

-----

### **How to Run the Project**

Follow these steps to set up and run the project on your local machine.

#### **Prerequisites**

Ensure you have Python installed. You can download it from the official Python website.

#### **Step 1: Clone the Repository**

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
```

#### **Step 2: Install Dependencies**

Install the required Python libraries using `pip`. It is recommended to use a virtual environment to manage project dependencies.

```bash
pip install pandas numpy scikit-learn flask
```

#### **Step 3: Run the Model Training Script**

The `carprediction.py` script will clean the data, train the model, and save the trained model as `Model.pkl`.

```bash
python carprediction.py
```

This will also generate a `cleaned_car.csv` file, which is used by the web application.

#### **Step 4: Run the Web Application**

Start the Flask application. This will launch a local web server.

```bash
python app.py
```

The application will run on `http://127.0.0.1:5000`. Open this URL in your web browser to access the car price prediction tool.

-----

### **Detailed Code Walkthrough**

#### **1. `carprediction.py`**

This script handles the entire machine learning pipeline.

  * **Data Loading:** It begins by loading the `quikr_car.csv` file into a Pandas DataFrame.
  * **Data Cleaning:** This is a crucial step where the script performs several transformations:
      * **Year Column:** It filters out non-numeric entries and converts the column to an integer type.
      * **Price Column:** It removes the "Ask For Price" entries, commas, and converts the column to an integer.
      * **`kms_driven` Column:** It cleans the column by removing 'Kms' and commas, then converts it to an integer.
      * **`fuel_type` Column:** It removes any rows with missing values (`NaN`).
      * **`name` Column:** It simplifies the car names by keeping only the first three words.
  * **Model Training:**
      * **Feature and Target Separation:** The cleaned dataset is split into `x` (features: `name`, `company`, `year`, `kms_driven`, `fuel_type`) and `y` (target: `Price`).
      * **Data Splitting:** The data is split into training and testing sets using `train_test_split`.
      * **Preprocessing Pipeline:** A `make_pipeline` is used to combine the `OneHotEncoder` and `LinearRegression` model. The `OneHotEncoder` is essential for converting categorical features like car name and company into a numerical format that the model can understand.
      * **Model Optimization:** The script runs the training process 1000 times with different random states to find the one that yields the highest **R² score**, a metric used to evaluate the model's performance. The final model is trained using this optimal random state.
  * **Model Saving:** The trained `pipe` (the entire pipeline) is saved as `Model.pkl` using `pickle`. This serialized object can be loaded later without retraining the model.

#### **2. `app.py`**

This script sets up the web application using Flask.

  * **Model Loading:** It loads the pre-trained `Model.pkl` file and the `cleaned_car.csv` data.
  * **Routes:**
      * **`/` route:** This is the home page. It serves the `index.html` template and passes the unique car companies and models to populate the dropdown menus on the web form.
      * **`/predict` route:** This route handles the form submission. It retrieves the user's input from the web form, creates a Pandas DataFrame with the input data, and uses the loaded model to make a prediction. The estimated price is then displayed on the `index3.html` template.

-----

### **Example Prediction**

A sample prediction can be made using the trained model with a new set of inputs.

**Input:**

  * **Name:** 'Maruti Suzuki Swift'
  * **Company:** 'Maruti'
  * **Year:** 2019
  * **Kms Driven:** 100
  * **Fuel Type:** 'Petrol'

**Output:**
The model would output an estimated price, for example, `[450000.0]`.

\!

-----

### **Contribute**

Contributions are welcome\! If you have any suggestions or improvements, please feel free to open a pull request or an issue on the GitHub repository.
