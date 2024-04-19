import keras
import nltk
import json
import pickle

nltk.download('punkt')
nltk.download('wordnet')

words = []
tags = []
documents = []
intents = open('intents.json').read()
intents = json.loads(intents)

lemmatizer = nltk.stem.WordNetLemmatizer()

for intent in intents:
    for pattern in intent['patterns']:
        pattern_words = nltk.word_tokenize(pattern)
        pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]
        words.extend(pattern_words)
        documents.append({'words': pattern_words, 'tag': intent['tag']})

        if intent['tag'] not in tags:
            tags.append(intent['tag'])

words = list(set(words))
tags = list(set(tags))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(tags, open('tags.pkl', 'wb'))

train_x = []
train_y = []
for doc in documents:
    x = []
    for w in words:
        x.append(1) if w in doc['words'] else x.append(0)

    train_x.append(x)
    train_y.append(tags.index(doc['tag']))

model = keras.models.Sequential()
model.add(keras.layers.Dense(128, input_shape=(len(words),), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(tags), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')

callback = keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True, monitor='loss', mode='min', verbose=1)

hist = model.fit(train_x, train_y, epochs=200, callbacks=callback)
