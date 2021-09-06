#import libraries
import numpy as np
import pandas as pd
from flask import Flask, render_template,request
import pickle as pkl
app = Flask(__name__)
model = pkl.load(open('Final_car.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    features_list = [x for x in request.form.values()]
    le_name = pkl.load(open('name_encoder.pkl','rb'))
    le_location = pkl.load(open('location_encoder.pkl','rb'))
    le_fuel_type = pkl.load(open('fuel_type_encoder.pkl','rb'))
    le_owner_type = pkl.load(open('owner_type_encoder.pkl','rb'))
    features_list[0] = le_name.transform(np.array(features_list[0]).reshape(-1,1))
    features_list[1] = le_location.transform(np.array(features_list[1]).reshape(-1,1))
    features_list[2] = float(features_list[2])
    features_list[3] = float(features_list[3])
    features_list[4] = le_fuel_type.transform(np.array(features_list[4]).reshape(-1,1))

    if features_list[5] == 'Manual':
        features_list[5] = 0 
    else:
        features_list[5] = 1

    features_list[6] = le_owner_type.transform(np.array(features_list[6]).reshape(-1,1))
    features_list[7] = float(features_list[7])
    features_list[8] = float(features_list[8])
    features_list[9] = float(features_list[9])
    features_list[10] = float(features_list[10])

    prediction = model.predict(np.array(features_list).reshape(-1,1))
    output = round(prediction[0], 2) 
    return render_template('index.html', prediction_text='Reselling Price of the vehicle is :{} lakhs'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
    