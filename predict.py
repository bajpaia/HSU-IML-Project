import pickle
import re
import nltk
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences

lemmatiser = nltk.stem.WordNetLemmatizer()
MODEL_PATH = './models/LSTM_One_HOT.json'
WEIGHT_PATH ='./models/LSTM_One_HOT.h5'
WORD_IDX_PATH = './models/word_index_classifer.pickle'
INT_TO_PERSON_PATH ='./models/int_to_label_one_hot.pickle'
INPUT_LENGTH = 50
STOPWORDS = set(stopwords.words('english'))
PERSONS = {'enfj', 'enfp', 'entj', 'entp', 'esfj', 'esfp', 
'estj', 'estp', 'infj', 'infp', 'intj', 'intp', 'isfj', 'isfp', 'istj', 'istp'}
nltk.download('all')


class Predictor:

    def __init__(self, MODEL_PATH = MODEL_PATH, WEIGHT_PATH = WEIGHT_PATH, INT_TO_PERSON_PATH= INT_TO_PERSON_PATH,WORD_IDX_PATH = WORD_IDX_PATH, INPUT_LENGTH = INPUT_LENGTH):
        
        self.model = model_from_json(open(MODEL_PATH).read())
        self.model.load_weights(WEIGHT_PATH)
        self.max_len = INPUT_LENGTH
        global graph
        graph = tf.get_default_graph() 
        pickle_in = open(WORD_IDX_PATH, 'rb')
        self.word_index = pickle.load(pickle_in)
        pickle_in = open(INT_TO_PERSON_PATH, 'rb')
        self.personalities = pickle.load(pickle_in)

    def preprocess_input(self, text, remove_stopwords = False):
        
        text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(text))
        text = re.sub("[^a-zA-Z]", " ", text)
        text = re.sub(' +', ' ', text).lower()

        if remove_stopwords:
            text = " ".join([lemmatiser.lemmatize(w) for w in text.split(' ') if w not in STOPWORDS and w not in PERSONS])
        
        else:
            text = " ".join([w for w in text.split(' ') if w not in PERSONS])
            
        return text

    def transform_input(self, input_text):
    
        input_text = self.preprocess_input(input_text, False)        
        seq = [self.word_index[word] for word in input_text.split() if word in self.word_index and self.word_index[word] <= 20000]
        seq = pad_sequences(np.array([seq]), self.max_len)
        
        return seq
    
    def predict_input_class(self, input_text):

        seq = self.transform_input(input_text)
        with graph.as_default():
            pred = self.model.predict([seq])

        prediction = self.personalities[np.argmax(pred[0])]
        print(pred)
        return (prediction, int(pred[0][np.argmax(pred)]*100))