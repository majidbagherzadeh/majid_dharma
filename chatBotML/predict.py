import nltk
import pickle
import numpy as np
import json
import random
from tkinter import *
import keras


model = keras.models.load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
tags = pickle.load(open('tags.pkl', 'rb'))


def clean_up_msg(msg):
    msg_words = nltk.word_tokenize(msg)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    msg_words = [lemmatizer.lemmatize(word.lower()) for word in msg_words]
    return msg_words


def predict_tag(msg_words):
    x = []
    for w in words:
        x.append(1) if w in msg_words else x.append(0)

    y = model.predict([x])[0]
    return np.argmax(y)


def get_response(pre_tag):
    for intent in intents:
        if intent['tag'] == tags[pre_tag]:
            return random.choice(intent['responses'])


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        msg_words = clean_up_msg(msg)
        pre_tag = predict_tag(msg_words)
        res = get_response(pre_tag)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", )

ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
