# Pradhan Mantri Fasal Beema Yojna Chatbot
The govenrnment of India's initiative Pradhan mantri Fasal Beema Yojna is a policy to give insurance to the farmers for their crops. There is a complete registeration process which may cause confusion. To solve this I created a chatbot which would help the users of the yojna in the registeration process if the face any issue.

![top_5_rows](/main screen.jpg)

# Data Acquisition:
The data that I used in the chatbot is custom-developed by me. I used a JSON file format to store the data, then load it into the python file using the JSON module. 
The json file has three main components.
<ul>
<li> Tag
<li> Patterns
<li> Responses
  </ul>
  
The "Tag" represents the specific tag or label associated.
The "Patterns" represents the type of text that may encounter related to the specific tag or label.
The "Response" represents the responses that the bot will give when that specific tag is encountered.
Together all three components enable the chatbot to analyze the sentiment of the text and give an appropriate response related to it.

# Data Preparation:

## Load in the data

After loading the json file into our project using the "json. load()" function, we need to prepare our data to organize the data in a format to train our model. Here we are organizing the data to form a pandas dataframe. To do this, we'll store each of our patterns and its respective tag into a list then create a dataframe using pd.DataFrame() function. Our dataframe will look like this.

![top_5_rows](/data.jpg)

We are also creating a dictionary to store respective responses related to the tags.

## Now we have our data loaded, the next step is to do some preprocessing.

First, we'll convert all our sentences to lower case and remove all the punctuation from the sentences. 

Since the model is trained only on integer data, we'll apply tokenization to our dataset. Tokenization is a process of assigning a word with an integer so that our sentence may be deduced in a numeric format. We'll set the max limit of these words to 2000.

In our dataset, we have sentences of different lengths of words, so we need to apply "padding" to our dataset, which means standardize our rows to a similar length. Padding is the process of adding zeros in front of all those rows having less fewer integers(words).

Lastly, we'll also apply label encoding to our tags as well using the sklearn preprocessing LabelEncoding() function.

# Modeling:- 

Our model will have four layers with an Input layer as our initial layer, next, there is an Embedding layer followed by an LSTM layer and a Dense layer as the final output layer with softmax activation. Let's discuss the layers used in-depth:-

<ul>
<li> Input:- It is used to instantiate a Keras tensor. We have a tensor created of the size of the input shape.
<li>Embedding layer:- An embedding layer turns positive integers (indexes) into dense vectors of fixed size. 
<li> LSTM:- LSTM stands for Long Short Term Memory is a built-in RNN layer in Tensorflow. An LSTM layer allows our network to persist information by looping over previous events. This adds to the accuracy of the model and the ability to make correct predictions.
<li> Dense:- The final layer of the model is the dense layer, We use softmax activation with the dimensions of our output shape so as to get as many nodes as there are tags.
</ul>
![top_5_rows](/main screen.jpg)
# Deployment:

After building and saving the model using the model.save() in a .h5 format, we'll use HTML and CSS to create a front end of our project. It uses aspects of JS as well to get the chatbot interphase appeal. Finally, we'll connect our front end with a back end. For this, we'll make use of the Flask framework. We'll create a function to handle the requests from the front end that will basically get the text using a GET request and preprocess that text as we did earlier(apply tokenization and padding then send to our model for prediction of the labels. Finally, our backend will send a valid response to our frontend using the dictionary we created earlier related to tags and responses. A random response will be sent to the front end out of all available responses. We'll have the interphase ready like this:-


[Go to the top](https://github.com/Pulkit12dhingra/PMFBY-chatbot)
