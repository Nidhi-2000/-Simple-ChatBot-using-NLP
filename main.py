import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

import random
with open("intents.json") as file:
    data = json.load(file)  //loads intents.json file
    


try:
    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)
except:


// extracting data
    words=[] 
    labels=[]
    docs_x=[]
    docs_y=[]

//loop through the json and extract data we want
//for each pattern we will turn into a list of words

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds= nltk.word.tokenize(pattern)  
        words.extend(wrds)
        docs_x.append(wrds)

docs_y.append(intent['tags'])
    
if intent['tags'] not in labels:
    labels.append(intent['tags'])
    
// WORD STEMMING
//stemming a word is attempting to find root of word
//it is used to reduce the vocubulary of our model

words = [stemmer.stem(w.lower()) for w in words if w!='?']
words = sorted(list(set(words)))

labels = sorted(labels)

//BAG OF WORDS
// Represent each sentence with a list the list will represent a word in our vocubulary

training=[]
output=[]

out_empty=[0 for _ in range(len(labels))]
for x,doc in enumerate(doc_x):
    bag=[]
    wrds=[stemmer.stem(w.lower()) for w in doc]
    
// if the position of the list is 1 then will mean that word exists else no it doesnt exist in our vocabulary

    for w in words:
        if w in wrds:
            bags.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]

output_row[labels.index(docs_y[x])]=1  // each position in the list represnt one distinct label

training.append(bag)
output.append(output_row)

// converting data and input to numpy array

training=numpy.array(training)
output=numpy.array(output)

with open("data(.pickle","wb") as f:
    pickle.dump((words,labels,training,output),f)
    
//defining the architecture for our model

tensorflow.reset_default_graph()

net=tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net=tflearn.regression(net)

model = tflearn.DNN(net)

//Training and Saving the model.

try:
    model.load("model.tflearn")
except:
    model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
    model.save('model.tflearn')
    
//Making prediction
// This function will transform string into bag of words
def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for s in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i]=1
    return numpy.array(bag)

// This function will handle getting prediction from the model

def chat():
    print("Start talking with the bot(type quit to stop)!")
    while True:
        inp = input("You:")
        if inp.lower()=='quit':
            break
        results = model.predict([bag_of_words(inp,words)])
        results_index=numpy.argmax(results)
        tag=labels[result_index]

        for tg in data["intents"]:
            if tg['tag']==tag:
                responses = tg['responses']

print(random.choice(responses))
chat()

// Now Run the program and enjoy chatting with the bot!!!

