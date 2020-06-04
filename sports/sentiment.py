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
import train
from train import find_feat,voteclass

l=[]
score=0


with open("new.txt") as file:
    for read in file:
        a=sent_tokenize(read)
        l.append(a)
        for i in a:
            analysis=TextBlob(i)
            print(i,analysis.sentiment.polarity)
            score=score+analysis.sentiment.polarity
    print(score)
    print("\n")

    l=[j.replace("\n","") for i in l for j in i]
    
    line=""
    for i in l:
        line=line+" "+str(i)

    anal=TextBlob(line)
    print(anal.sentiment.polarity)
    feat=find_feat(line)
        
    print("classification : ",voteclass.classify(feat),"confidence :",
      voteclass.confidence(feat))
        
    score=0
    line=""
    l=[]


