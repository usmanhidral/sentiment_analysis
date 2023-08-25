import numpy as np
import pandas as pd
import io
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs
from sklearn import model_selection
from sklearn import metrics
import torch
import torch.nn as nn
import tensorflow as tf 

torch.manual_seed(1024);



model =  torch.load("Model/model")

def Interact_user_input(model):
    '''
    model: trained model : fasttext model or glove model
    '''
    model.eval()
    
    sentence = ''
    while True:
        try:
            sentence = input('Review: ')
            if sentence in ['q','quit']: 
                break
            sentence = np.array([sentence])
            sentence_token = tokenizer.texts_to_sequences(sentence)
            sentence_token = tf.keras.preprocessing.sequence.pad_sequences(sentence_token, maxlen = MAX_LEN)
            sentence_train = torch.tensor(sentence_token, dtype = torch.long).to(device, dtype = torch.long)
            predict = model(sentence_train)
            if predict.item() > 0.5:
                print('------> Positive')
            else:
                print('------> Negative')
        except KeyError:
            print('please enter again')
    
    
    


Interact_user_input(model)
