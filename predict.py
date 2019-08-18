import pickle
import re
import nltk
import numpy as np
# import tensorflow as tf
# from keras.models import model_from_json
# from nltk.corpus import stopwords

# lemmatiser = nltk.stem.WordNetLemmatizer()
# MODEL_PATH = './models/precious_classifier_v1.json'
# WEIGHT_PATH ='./models/precious_classifier_v1.h5'
# WORD_IDX_PATH = './models/word_index_classifer.pickle'
# INPUT_LENGTH = 300
# STOPWORDS = set(stopwords.words('english'))
# PERSONS = {'enfj', 'enfp', 'entj', 'entp', 'esfj', 'esfp', 
# 'estj', 'estp', 'infj', 'infp', 'intj', 'intp', 'isfj', 'isfp', 'istj', 'istp'}
# personalities = {2 : 'creative',
#               1:'articulate',
#               4:'zen',
#              3:'curious',
#              5:'decisive'}


class Predictor:

    # def __init__(self, MODEL_PATH = MODEL_PATH, WEIGHT_PATH = WEIGHT_PATH, WORD_IDX_PATH = WORD_IDX_PATH, INPUT_LENGTH = INPUT_LENGTH):
        
    #     self.model = model_from_json(open(MODEL_PATH).read())
    #     self.model.load_weights(WEIGHT_PATH)
    #     self.max_len = INPUT_LENGTH
    #     global graph
    #     graph = tf.get_default_graph() \
    #     pickle_in = open(WORD_IDX_PATH, 'rb')
    #     self.word_index = pickle.load(pickle_in)

    def __init__(self):
        pass

    
    def predict_class(self, text):

        return ("INFJ", 90)