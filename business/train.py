# importing packages
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
# loading all the english stopwords
stop_words=set(stopwords.words("english"))

# creating a class which will be performing the sentiment classification of the document.
# it takes multiple classifiers as the parameter
class voteclassifier(ClassifierI):
     def __init__(self,*classifiers):
          self._classifiers=classifiers
    # this function uses all the classifiers passed as the argument while creating the object of the class
    # and predict the sentiment of the document using all the passed class
    # the majority output from all the classes is taken as the result and returned
     def classify(self,features):
          votes=[]

          for c in self._classifiers:
               v=c.classify(features)
               votes.append(v)
          return mode(votes)

    # this function is used to find out, among all the used classifiers how many classifiers agree on the same result
     def confidence(self,features):
          votes=[]

          for c in self._classifiers:
               v=c.classify(features)
               votes.append(v)

          choice_votes=votes.count(mode(votes))
          conf=choice_votes/len(votes)
          return conf


# here we are loading the object wihch we stored after data preprocessing
with open("business.pickle",'rb') as dic:
     business=pickle.load(dic)
     
doc=[]
pos=[]
neg=[]
pos_words=[]
neg_words=[]
# here all the words from the positively classifier training document 
# and negatively classifier training document are stored in respective variables.
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
# pool of all the available words are being created here 
for i in pos_words:
     all_words.append(i)

for i in neg_words:
     all_words.append(i)

# we are finding the frequence distribution of our corpus
all_words=nltk.FreqDist(all_words)

words=list(all_words.keys())[:5000]

# this function is used to vectorize the given document based on the all the words that are present in our word pool/word corpus.
def find_feat(document):
    word=word_tokenize(document)
    feat={}
    for w in words:
        feat[w]=(w in word)

    return feat

# we are vectorizing all the training documents here by accessing the  binary file which contains the preprocessed dataset.
feature_Sets=[(find_feat(rev),category) for (rev,category) in doc]

random.shuffle(feature_Sets)

# splitting training and testing dataset.
train=feature_Sets[:300]
test=feature_Sets[300:]

# from here on a series of 6 classifiers are created and trained on the training dataset and 
# evaluated on the testing dataset
# these classifiers are used for enssemble learning 
classifier=nltk.NaiveBayesClassifier.train(train)
print("\noriginal naive bayes accuracy:",nltk.classify.accuracy(classifier,test))
classifier.show_most_informative_features(15)

mnb_classifier=SklearnClassifier(MultinomialNB())
mnb_classifier.train(train)
print("\nmnb_classifier accuracy:",nltk.classify.accuracy(mnb_classifier,test))


bnb_classifier=SklearnClassifier(BernoulliNB())
bnb_classifier.train(train)
print("\nbnb_classifier accuracy:",nltk.classify.accuracy(bnb_classifier,test))

logr_classifier=SklearnClassifier(LogisticRegression())
logr_classifier.train(train)
print("\nlogr_classifier accuracy:",nltk.classify.accuracy(logr_classifier,test))


sgd_classifier=SklearnClassifier(SGDClassifier())
sgd_classifier.train(train)
print("\nsgd_classifier accuracy:",nltk.classify.accuracy(sgd_classifier,test))


linearsvc_classifier=SklearnClassifier(LinearSVC())
linearsvc_classifier.train(train)
print("\nlinearsvc_classifier accuracy:",nltk.classify.accuracy(linearsvc_classifier,test))


svc_classifier=SklearnClassifier(SVC())
svc_classifier.train(train)
print("\nsvc_classifier accuracy: ",nltk.classify.accuracy(svc_classifier,test))

# we create an object of the above created class voteclassifier and pass all the created classifier object as parameters
voteclass=voteclassifier(classifier,
                         mnb_classifier,
                         bnb_classifier,
                         linearsvc_classifier,
                         sgd_classifier,
                         svc_classifier,
                         logr_classifier)


print("\nAccuracy of The model: ",nltk.classify.accuracy(voteclass,test))                     


# print("classification : ",voteclass.classify(test[0][0]),"confidence :",
#       voteclass.confidence(test[0][0]))

##save_classifier = open("naivebayer.pickle",wb")
##pickle.dump(classifier,save_classifier)
##save_classifier.close()
