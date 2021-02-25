from flask import Flask, render_template, request
#importing the libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from gevent.pywsgi import WSGIServer

import random

app = Flask(__name__)


with open('intents.json') as content:
	data1 = json.load(content)
#getting all the data to lists
tags = []
inputs = []
responses={}
for intent in data1['intents']:
 	responses[intent['tag']]=intent['responses']
 	for lines in intent['patterns']:
 		inputs.append(lines)
 		tags.append(intent['tag'])
#converting to dataframe
data = pd.DataFrame({"inputs":inputs,
                     "tags":tags})

#print(data)


#removing punctuations
import string
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))
#tokenize the data
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

#apply padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)

#encoding the outputs
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])


#input length
input_shape = x_train.shape[1]
#print(input_shape)
#define vocabulary
vocabulary = len(tokenizer.word_index)
#output length
output_length = le.classes_.shape[0]
#print("output length: ",output_length)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
	userText = request.args.get('msg')
	#print(userText)
	model = tf.keras.models.load_model('my_model.h5')
	texts_p = []
	prediction_input = userText
	#removing punctuation and converting to lowercase
	prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
	prediction_input = ''.join(prediction_input)
	texts_p.append(prediction_input)
	#tokenizing and padding
	prediction_input = tokenizer.texts_to_sequences(texts_p)
	prediction_input = np.array(prediction_input).reshape(-1)
	prediction_input = pad_sequences([prediction_input],input_shape)
	#getting output from model
	output = model.predict(prediction_input)
	output = output.argmax()
	#finding the right tag and predicting
	response_tag = le.inverse_transform([output])[0]

	
	return str(random.choice(responses[response_tag]))


if __name__ == "__main__":
    #app.run()
	http_server = WSGIServer(('localhost:5000'), app)
	http_server.serve_forever()