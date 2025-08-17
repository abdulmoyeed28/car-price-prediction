from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load the model
model = pickle.load(open('Model.pkl', 'rb'))
data = pd.read_csv('cleaned_car.csv')
app = Flask(__name__)

@app.route('/')
def home():
    companies = sorted(data['company'].unique()) 
    car_models = sorted(data['name'].unique())

    return render_template('index.html', companies=companies, car_models=car_models)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            name = request.form['name']
            company = request.form['company']
            year = int(request.form['year'])
            kms_driven = int(request.form['kms_driven'])
            fuel_type = request.form['fuel_type']
            
            input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]], 
                                      columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
            
            prediction = model.predict(input_data)
            
            return render_template('index3.html', prediction_text='Estimated Car Price: ₹{:.2f}'.format(prediction[0]))
        
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
