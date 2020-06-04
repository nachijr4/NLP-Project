# in this file we are preprocessing all the data and we are creating a labelled dataset.
# importing all the required packages
from news import senti
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
import string
import csv
import sys
import pickle
from textblob import TextBlob,Word
from nltk import ngrams
import re
from textblob.sentiments import NaiveBayesAnalyzer


l1=[]

business={}
# naming each file properly from number i to 511
for i in range(1,511):
    if i<10:
        k="0"+"0"+str(i)
    elif i<100:
        k="0"+str(i)
    else:
        k=str(i)
    l1.append(k)


l=[]
# initializing required variables
count=0
count1=0
count2=0
# reading each txt file in the "datasets/bbc/business/" folder
score=0
for i in l1:
    with open("datasets/bbc/business/"+str(i)+".txt") as file:
        for read in file:
            # sentence tonenizing the opened document
            a=sent_tokenize(read)
            l.append(a)
            for i in a:
                # getting a sentiment score for each sentence in the opened text
                # the output score is between -1 and 1 , denoting negative and possitive respectively
                analysis=TextBlob(i)
                # adding the score of each sentence to a variable in order to find the cummulative score of the document
                score=score+analysis.sentiment.polarity

        l=[j.replace("\n","") for i in l for j in i]
        # joining the tokenized sentences and forming the complete string again
        line=""
        for i in l:
            line=line+" "+str(i)
        # based on the cumulative score of all the sentences in that particular document, 
        # the sentiment of the document is assigned
        # storing the content of each the text document and its sentiment as key value pair respectively in a single dictionary.
        if score>0:
            count=count+1
            business[line]=1
        elif score<0:
            count1=count1+1
            business[line]=-1
        else:
            count2=count2+1
            business[line]=0
            
        score=0
        line=""
        l=[]

print(count,count1,count2)

# the dictionary wihch contains the content of each document as key and its sentiment as value is stored in binary format
with open("business1.pickle",'wb') as dic:
     pickle.dump(business,dic,protocol=pickle.HIGHEST_PROTOCOL)

