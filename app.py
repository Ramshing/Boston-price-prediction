from flask import Flask,request,app,jsonify,url_for,render_template
import json
import pickle
import numpy as np
import pandas as pd
import ast

## WSGI application - intermediate between web server and web app
app=Flask(__name__)

#Load the model
scaler=pickle.load(open('scaling.pkl','rb'))
regmodel=pickle.load(open('regmodel.pkl','rb'))


@app.route('/')   # Decorator
def home():
    return render_template("index.html")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    #arr=list(data.values())
    #column_vector = [item for item in arr]
    #reshaped_list = [column_vector]
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    #new_data=scaler.transform(np.array(json_array))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0]),200


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("result.html",result=output)

if __name__=='__main__':
    app.run(debug=True)