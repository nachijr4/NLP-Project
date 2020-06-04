# Ensemble Learning Sentiment Analysis

In this project we create an ensemble learning ML model to predict the sentiment (Positive/Negative) of a news article.
We have 3 Main files
- bbc.py
- train.py
- sentiment.py


#### Bbc.py
This file is used to label the dataset that we use to train our model. Data was collected form the BBC news dataset. This dataset contains articles published by BBC organised into categories like, business, sport, entertainment etc.
In this file we loop through each article and pass it to textblob. Textblob will give us the sentiment polarity for each sentence which will add up to the sentiment polarity value of the article. The article text is stored in a dictionary as the key and the label which we get from the polarity will be the value.
This dictionary will be stored as a .pickle file using the pickle module

#### Train.py
This file trains our prediction model with the labelled dataset.
Here we have defined a class voteclassifier that will take 7 trained prediction models as an input.
We extract the previously stored dictionary from the .pickle file. The code will loop through the dictionary and the text of the article will be converted to a vectorised from using the find_feat function. A list is created that contains every article in the from of a tuple of the vectorised text and the label of the article.
This list is now split into the train and test dataset.
Each individual model (NaiveBayesClassifier, MultinomialNB, BernoulliNB, LogisticRegression, SGDClassifier, LinearSVC, SVC) is trained using the training set. The accuracy of each model is also checked using the test data.
All 7 models will be passed to the voteclassifier class. This class will call the classify function for each model and return the mode of all the classifications. This will be the final prediction of our model.

#### Sentiment.py
This file will show the final prediction for an unseen new article. 
It opens the testdata.txt file and classifies the text of the article and displays the result along with the confident. In order to test this output we will also run the text through the textblob pipeline.
Both outputs will be displayed and can be compared.

###Steps for executing the project
First the bbc.py file has to be run so that is will process the bbc dataset and create a labelled dataset which can be used for training out ensemble model
then the sentiment.py file inside business folder, has to be run, that file will in turn execute the train.py file and classify the document given.
