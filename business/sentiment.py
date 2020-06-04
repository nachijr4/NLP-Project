# this is the master file while utilizes all the function present in the train.py file.
# here we process the test data and classify it using our created model which uses enssemble learning
# and we classify it using textblob, in the end we check how accurate our model is

# import packages
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle
from textblob import TextBlob,Word
from textblob.sentiments import NaiveBayesAnalyzer
from train import find_feat,voteclass

l=[]
score=0

# opening the text file that needs to be classified.
with open(".\testdata.txt") as file:
    for read in file:
        # the file is sentence tokenized.
        a=sent_tokenize(read)
        l.append(a)
        # we are find the sentiment polarity for each sentence
        # we add all the sentence scores to get the final score of the document.
        for i in a:
            analysis=TextBlob(i)
            print(i,"(",analysis.sentiment.polarity,")\n")
            score=score+analysis.sentiment.polarity


# the output for the textblob classification is printed here
    textblob_result = ""
    if score>0:
        textblob_result = "Positive"
    else:
        textblob_result = "Negative"
    print("\n\n-------------------------")    
    print("Sentiment Analysis of textblob: ",textblob_result)
    print("-------------------------\n")
    

    l=[j.replace("\n","") for i in l for j in i]
    # joining the tokenized string and making it as a single string again
    line=""
    for i in l:
        line=line+" "+str(i)
    # finding the feature vector for the prediction data.
    feat=find_feat(line)
    # classifying the prediction data using the classify function present in the voteclass Class object.
    # this function utilizes all the 6 models we have created in train.py and gives the output
    result = ""
    if voteclass.classify(feat)=="pos":
        result = "Positive"
    else:
        result = "Negative"
# the result of the prediction by our model and the confidence of the model is printed 
    print("\n-------------------------")
    print("Our model's performance\n")
    print("Classification : ",result,"\nConfidence :",voteclass.confidence(feat))
    print("-------------------------\n")


