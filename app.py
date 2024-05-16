#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[3]:


app = Flask(__name__)
model = pickle.load(open('Alert.pkl', 'rb'))


# In[4]:


@app.route('/')
def home():
    return render_template('template.html')


# In[5]:


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    prediction_text = "Generate Alert" if output == 1 else "No Alert"

    return render_template('index.html', prediction_text=prediction_text)


# In[19]:


if __name__== "__main__":
    app.run(debug=True)






