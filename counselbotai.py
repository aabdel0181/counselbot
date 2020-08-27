import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

#supress depreciation messages (current tf bug)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

#load in data 
with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

#word processing 
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

#Creating Neural Network Model

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#Fit the Model
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

#Deploy 
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

newQuestions = []

def chat():
    print("Start talking with the CounselBot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            print("CounselBot: Goodbye!!")
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        #weeds out gibbgerish responses
        if results[results_index] > 0.7:

            #print(results) <<<< allows you to see the score of each possible resposne; the highest one is what is displayed
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print("CounselBot: "+random.choice(responses))
        else: 
            print("CounselBot: Sorry, I didn't understand that. I am adding your request to my neural network database to be updated. Please ask another question")
            newQuestions.append(inp)
print("CounselBot: Hello and welcome to CounselBot.Ai")
print("CounselBot: You can discuss mental health, ask me about schedules or any of your academic or personal inquiries/concerns!")
chat()
numpy.savetxt("newQuestions.txt", numpy.array(newQuestions), fmt="%s")