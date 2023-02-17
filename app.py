from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import requests

dataset=pd.read_csv('dataset/dataset.csv')

app = Flask(__name__) 

nb_model=pickle.load(open('model/nb_model.pkl','rb'))
rf_model=pickle.load(open('model/rf_model.pkl','rb'))
svm_model=pickle.load(open('model/svm_model.pkl','rb'))

symptoms = dataset.iloc[:,:-1].columns.values
le =LabelEncoder()
dataset['prognosis'] = le.fit_transform(dataset['prognosis'].values)
symptom_index = {}
data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":le.classes_
}

input_data = [0] * 132
dict={}
for i in range(0,132):
    dict[symptoms[i]]=i
    
label=1
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/diseaseprediction')
def disease():
    return render_template('diseaseprediction.html')

@app.route('/diseaseprediction/predict',methods=['POST'])
def predict():
    sym1=request.form.get('Symptom_1')
    sym2=request.form.get('Symptom_2')
    sym3=request.form.get('Symptom_3')
    sym4=request.form.get('Symptom_4')
    sym5=request.form.get('Symptom_5')
    global input_data
    input_data[dict[sym1]]=1
    input_data[dict[sym2]]=1
    input_data[dict[sym3]]=1
    input_data[dict[sym4]]=1
    input_data[dict[sym5]]=1
    input_data=np.array(input_data).reshape(1,-1)
    print(input_data)
    #return data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
    return render_template('diseaseprediction.html',label=data_dict["predictions_classes"][rf_model.predict(input_data)[0]])
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/plan')
def plan():
    return render_template('plan.html')
    
@app.route('/plan/result',methods=['POST'])
def result():
    key='cf0358801e80d8542089c2334f5803c2'
    city1=request.form.get('cityName')
    city2=request.form.get('cityName2')
    api1 = "https://api.openweathermap.org/data/2.5/weather?q=" + city1 + "&appid=" + key + "&units=metric";
    api2 = "https://api.openweathermap.org/data/2.5/weather?q=" + city2 + "&appid=" + key + "&units=metric";
    co1=requests.get(api1)
    co2=requests.get(api2)
    lon1=co1.json()['coord']['lon']
    lat1=co1.json()['coord']['lat']
    lon2=co2.json()['coord']['lon']
    lat2=co2.json()['coord']['lat']
    temp1=co1.json()['main']['temp']
    temp2=co2.json()['main']['temp']
    des1=co1.json()['weather'][0]['description']
    des2=co2.json()['weather'][0]['description']
    api3="https://api.openweathermap.org/data/2.5/air_pollution?lat=" + str(lat1) + "&lon=" + str(lon1) + "&appid=" + key
    api4="https://api.openweathermap.org/data/2.5/air_pollution?lat=" + str(lat2) + "&lon=" + str(lon2) + "&appid=" + key
    api3=requests.get(api3)
    api4=requests.get(api4)
    aqi1=api3.json()['list'][0]['main']['aqi']
    aqi2=api4.json()['list'][0]['main']['aqi']
    cond1='Weather at {} is {}'.format(city1,des1)
    cond2='Weather at {} is {}'.format(city2,des2)
    temp1='Temperature at {} is {}'.format(city1,temp1)
    temp2='Temperature at {} is {}'.format(city2,temp2)
    if aqi2>aqi1:
        label1='The air is more contaminated at {}'.format(city2)
    else:
        label1='The air is comparatively safe. Safe to travel :)'
    
    return render_template('plan.html',label=[cond1,cond2,temp1,temp2,label1])
    



if __name__=='__main__':
     app.run(debug=True)