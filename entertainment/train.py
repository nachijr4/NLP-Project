import nltk
import string
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
import pickle
from nltk.classify import ClassifierI
from statistics import mode
import re

stop_words=set(stopwords.words("english"))

class voteclassifier(ClassifierI):
     def __init__(self,*classifiers):
          self._classifiers=classifiers

     def classify(self,features):
          votes=[]

          for c in self._classifiers:
               v=c.classify(features)
               votes.append(v)
          return mode(votes)
     def confidence(self,features):
          votes=[]

          for c in self._classifiers:
               v=c.classify(features)
               votes.append(v)

          choice_votes=votes.count(mode(votes))
          conf=choice_votes/len(votes)
          return conf


with open("entertainment.pickle",'rb') as dic:
     business=pickle.load(dic)
     
doc=[]
pos=[]
neg=[]
pos_words=[]
neg_words=[]

for i in business:
    if business[i]==1:
        doc.append((i,"pos"))
        pos.append(i)
    elif business[i]==-1:
        doc.append((i,"neg"))
        neg.append(i)


for i in pos:
    a=word_tokenize(i)
    for word in a:
        word=word.lower()  
        if word not in stop_words:
            if word not in string.punctuation:
                if word.isalpha():
                    if re.match("(g\w+W+)","hello")!="None":
                        pos_words.append(word)
for i in neg:
    a=word_tokenize(i)
    for word in a:
        word=word.lower()  
        if word not in stop_words:
            if word not in string.punctuation:
                if word.isalpha():
                    if re.match("(g\w+W+)","hello")!="None":
                        neg_words.append(word)
        

all_words=[]

for i in pos_words:
     all_words.append(i)

for i in neg_words:
     all_words.append(i)


all_words=nltk.FreqDist(all_words)

words=list(all_words.keys())[:5000]


def find_feat(document):
    word=word_tokenize(document)
    feat={}
    for w in words:
        feat[w]=(w in word)

    return feat


feature_Sets=[(find_feat(rev),category) for (rev,category) in doc]

random.shuffle(feature_Sets)


train=feature_Sets[:250]
test=feature_Sets[250:]


classifier=nltk.NaiveBayesClassifier.train(train)
print("original naive bayes",nltk.classify.accuracy(classifier,test))
classifier.show_most_informative_features(15)

mnb_classifier=SklearnClassifier(MultinomialNB())
mnb_classifier.train(train)
print("mnb_classifier",nltk.classify.accuracy(mnb_classifier,test))


bnb_classifier=SklearnClassifier(BernoulliNB())
bnb_classifier.train(train)
print("bnb_classifier",nltk.classify.accuracy(bnb_classifier,test))

##gnb_classifier=SklearnClassifier(GaussianNB())
##gnb_classifier.train(train.toarray())
##print("gnb_classifier",nltk.classify.accuracy(gnb_classifier,test))

logr_classifier=SklearnClassifier(LogisticRegression())
logr_classifier.train(train)
print("logr_classifier",nltk.classify.accuracy(logr_classifier,test))


sgd_classifier=SklearnClassifier(SGDClassifier())
sgd_classifier.train(train)
print("sgd_classifier",nltk.classify.accuracy(sgd_classifier,test))


linearsvc_classifier=SklearnClassifier(LinearSVC())
linearsvc_classifier.train(train)
print("linearsvc_classifier",nltk.classify.accuracy(linearsvc_classifier,test))


svc_classifier=SklearnClassifier(SVC())
svc_classifier.train(train)
print("svc_classifier",nltk.classify.accuracy(svc_classifier,test))


voteclass=voteclassifier(classifier,
                         mnb_classifier,
                         bnb_classifier,
                         linearsvc_classifier,
                         sgd_classifier,
                         svc_classifier,
                         logr_classifier)

print("voteclass",nltk.classify.accuracy(voteclass,test))                     


print("classification : ",voteclass.classify(test[0][0]),"confidence :",
      voteclass.confidence(test[0][0]))

##save_classifier = open("naivebayer.pickle",wb")
##pickle.dump(classifier,save_classifier)
##save_classifier.close()
