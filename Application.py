#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server

import joblib
import numpy as np

model = joblib.load('diabetes_classifier_model.pkl')
scaler=joblib.load('Scaler.pkl')

app = Flask(__name__)


def bmi_app():
    name=input("Whats Your name")
    put_text('Hi, welcome to our website', name)
    
    
    preg=input('How many pregnancies did you have?(0 if you are male)', type=NUMBER)
    glucose=input('What is your Fasting Glucose level?',  type=NUMBER)
    bp=input('What is your Blood Pressure?', type=NUMBER)
    age=input('What is your age?',type=NUMBER)
    weight=input('What is your weight?(In Kilograms)',type=NUMBER)
    height=input('What is your height?(In centimeters)',type=NUMBER)
    history=select('Does any of your parent have diabetes?', ['YES', 'NO'])
    if history=='YES':
        history=1
    else:
        history=0
        
    bmi=weight/(height/100)**2
    
    prediction=model.predict(scaler.transform([[preg,glucose,bp,bmi,age,history]]))
    
    if prediction[0]==1:
        put_text("You have risk of getting Type 2 Diabetes")
    else:
        put_text("You do not have risk of getting Type 2 Diabetes")
        
    
    
    
        
app.add_url_rule('/tool', 'webio_view', webio_view(bmi_app),
            methods=['GET', 'POST', 'OPTIONS'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(bmi_app, port=args.port)


# In[ ]:




