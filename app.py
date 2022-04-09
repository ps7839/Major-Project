# -*- coding: utf-8 -*
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('finalmodel.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output:
        return render_template('index.html', prediction_text='Predicted Output = {}'.format(output))
    else:
        return render_template('index.html', prediction_text='Predicted Output = {}'.format(output))

@app.route('/predict_api',methods=['GET','POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output == 1:
        return "HELLO"
    else:
         return "BYE"


if __name__ == "__main__":
    app.run(debug=True)
    
#https://mighty-tundra-22355.herokuapp.com
