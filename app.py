from flask import Flask, render_template, url_for, request
import sqlite3


import pickle
import numpy as np
import random
import requests
import warnings
warnings.filterwarnings('ignore')


import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
intents = json.loads(open('intents1.json').read())
words = pickle.load(open('words.pkl','rb'))
labels = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence):
    cbmodel = load_model('chatbot_model1.h5')
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = cbmodel.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list

#getting chatbot response
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text)
    res = getResponse(ints, intents)
    return res



import shutil
import os
import sys
import json
import math
from PIL import Image
from flask import Flask, render_template, request
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from keras.models import load_model
import random




connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()
command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            data=requests.get("https://api.thingspeak.com/channels/2191752/feeds.json?api_key=UV4HYQEF6E65N4E8&results=2")
            hb=float(data.json()['feeds'][-1]['field2'])
            temp=float(data.json()['feeds'][-1]['field1'])
            oxy=float(data.json()['feeds'][-1]['field3'])
            bp=float(data.json()['feeds'][-1]['field4'])
            return render_template('userlog.html',hb=hb,temp=temp,oxy=oxy,bp=bp)

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')


@app.route("/predictml", methods = ['POST', 'GET'])
def predictml():
    if request.method == 'POST':
        name = request.form['name']
        Age = request.form['age']
        sex = request.form['sex']
        # bp = request.form['bp']
        oxy = request.form['oxy']
        hb = request.form['heart']
        # ecg = request.form['ecg']
        Temperature = request.form['Temperature']
        to_predict_list = np.array([[oxy,hb,Temperature]])
        mlmodel = pickle.load(open('new.pkl', 'rb'))
        prediction = mlmodel.predict(to_predict_list)[0]
        print("machine learning prediction {}  :  ".format(prediction))
        return render_template('chatbot.html', prediction=prediction)
    return render_template('userlog.html')

@app.route("/predictcb", methods = ['POST', 'GET'])
def predictcb():
    if request.method == 'POST':
        query = request.form['query']
        ml = int(request.form['ml'])
        res = chatbot_response(query)

        chatbot_pred = 'gg'
        if 'mild' in res:
            chatbot_pred = 'NPCR'
        if 'moderate' in res:
            chatbot_pred = 'NFI'
        if 'severe' in res:
            chatbot_pred = 'PCR'

        if chatbot_pred == 'gg':
            return render_template('chatbot.html', prediction=ml, res=res)
        else:
            print('chatbot prediction {}'.format(chatbot_pred))

            # ml = 1 : Healthy
            # ml = 2 : not sure
            # ml = 3 : Covid detected

            if ml == 1 and chatbot_pred == 'NPCR':
                prediction = 1
                result = 'Sensor = Healthy, chatbot_pred = NPCR '
                instruction = 'You have no possible covid risk, Good Luck!!'
            if ml == 1 and chatbot_pred == 'NFI':
                prediction = 2
                result = 'Sensor = Healthy, chatbot=NFI'
                instruction = 'There is need for further investigation'
            if ml == 1 and chatbot_pred ==  'PCR':
                prediction = 3
                result = 'Sensor = Healthy , chatbot=PCR '
                instruction = 'Possible covid risk'

            if ml == 2 and chatbot_pred == 'NPCR':
                prediction = 4
                result = 'Sensor = Not Sure,  chatbot_pred = NPCR  '
                instruction = 'ssssssssssssssssss'
            if ml == 2 and chatbot_pred == 'NFI':
                prediction = 5
                result = 'Sensor = Not Sure, chatbot=NFI'
                instruction = 'ssssssssssssssssss'
            if ml == 2 and chatbot_pred ==  'PCR':
                prediction = 6
                result = 'Sensor = Not Sure, chatbot=PCR'
                instruction = 'ssssssssssssssssss'
            
            if ml == 3 and chatbot_pred == 'NPCR':
                prediction = 7
                result = 'Sensor = Critical, chatbot_pred = NPCR  '
                instruction = 'ssssssssssssssssss'
            if ml == 3 and chatbot_pred == 'NFI':
                prediction = 8
                result = 'Sensor = Critical, chatbot = NFI'
                instruction = 'ssssssssssssssssss'
            if ml == 3 and chatbot_pred ==  'PCR':
                prediction = 9
                result = 'Sensor = Critical, chatbot = PCR'
                instruction = 'ssssssssssssssssss'

            print('ml and chatbot prediction: {}'.format(prediction))
            print('result: {}'.format(result))
            print('instruction: {}'.format(instruction))
            return render_template('image.html', prediction=prediction, result=result, instruction=instruction)
    return render_template('chatbot.html')

@app.route("/predictip", methods = ['POST', 'GET'])
def predictip():
    if request.method == 'POST':
        dirPath = "static/images"
        fileName=request.form['filename']
        cb = int(request.form['cb'])
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'healthyvsunhealthynew-{}-{}.model'.format(LR, '2conv-basic')
        def process_verify_data():
            verifying_data = []
            path = os.path.join("static/test/"+fileName)
            img_num = fileName.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        tf.compat.v1.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 3, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)

        for num, data in enumerate(verify_data):
            img_num = data[1]
            img_data = data[0]
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            model_out = model.predict([data])[0]
            pred = np.argmax(model_out)
            print('cnn prediction {}'.format(pred))
        prediction = 0
        result = 'gg'
        instruction = 'gg'
        if cb == 1 and pred == 0:
            prediction = 0
            result = 'cb == 1 and pred == 0 '
            instruction = 'ssssssssssssssssss'
        if cb == 1 and pred == 1:
            prediction = 1
            result = 'cb == 1 and pred == 1'
            instruction = 'ssssssssssssssssss'
        if cb == 1 and pred == 2:
            prediction = 2
            result = 'cb == 1 and pred == 2'
            instruction = 'ssssssssssssssssss'
        if cb == 2 and pred == 0:
            prediction = 0
            result = 'cb == 2 and pred == 0'
            instruction = 'ssssssssssssssssss'
        if cb == 2 and pred == 1:
            prediction = 1
            result = 'cb == 2 and pred == 1'
            instruction = 'ssssssssssssssssss'
        if cb == 2 and pred == 2:
            prediction = 2
            result = 'cb == 2 and pred == 2'
            instruction = 'ssssssssssssssssss'
        if cb == 3 and pred == 0:
            prediction = 0
            result = 'cb == 3 and pred == 0'
            instruction = 'ssssssssssssssssss'
        if cb == 3 and pred == 1:
            prediction = 1
            result = 'cb == 3 and pred == 1'
            instruction = 'ssssssssssssssssss'
        if cb == 3 and pred == 2:
            prediction = 2
            result = 'cb == 3 and pred == 2 '
            instruction = 'ssssssssssssssssss'
        
        print('final prediction {}'.format(prediction))
        print('result: {}'.format(result))
        print('instruction: {}'.format(instruction))
        
        return render_template('image.html', result1=result, instruction1=instruction)
    return render_template('image.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
