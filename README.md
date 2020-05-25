# NLP Classification
## Summary
In this project I will use the 20 Newsgroups dataset to train a multiclass text classification model that will accuractely classify newsgroups postings from one of these 20 groups into the correct class. The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics.

<h5 align="center">20 Newsgroups</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/wordcloud.png" width=850>
</p>

## Data
I obtained the dataset from the Scikit-Learn library. This data set come pre-packaged in Scikit-Learn, already split into train/test and with the text, target and target name as variables. However, the target_name column in this dataset is incomplete and corrutped. To overcome this, I found an older json source of the same dataset online and us
ed it to fill in the missing values.

<h5 align="center">Missing target_names</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/target_names.png" width=850>
</p>

## Data Preprocessing 
This process consisted of 4 main steps: removal of stop words, tokenisation, lemmatisation and TF-IDF vectorisation. This was achieved using the nltk library. 

`Stop words` commonly used words in any language, e.g. 'a', 'the', 'is' in English. The purpose of identifying them is so that they can then be removed from the text so that the classifiers can focus on important words instead.  In addition to the standard stop words included in the nltk library, I also added some stop words specific to this corpus such as 'post', 'host' and 'subject. 

`Tokenisation` is the process of breaking text up into little pieces, or tokens. I used a RegEx tokeniser to make each individual word or number a token, and also removing all punctuation, as punctuation is not useful for classification.

`Lemmatisation` is the process of transforming each word into its root form. E.g. the word 'walk' could appear as 'walk', 'walking', 'walked','walks' etc. These all have the same meaning but will appear different to a computer. Lemmatisation will transform all of these into 'walk'.

`TF-IDF` (short for term frequency–inverse document frequency) is the final step.  TF-IDF is intended to reflect how relevant a term is in a given document, and gives a numerical score that can be input into a machine learning model. 

This workflow is visualised below:

<h5 align="center">Raw Text</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/rawtext.png" width=850>
</p>

<h5 align="center">Removal of Stopwords & Tokenisation</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/tokenisation.png" width=850>
</p>

<h5 align="center">Lemmatisation</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/lemmatisation.png" width=850>
</p>

<h5 align="center">TF-IDF</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/tfidf.png" width=850> 
</p>

## Part of Speech Tagging
Part of speech (pos) tagging is the process of marking each word with a label that indicates what part of speech it is, i.e. noun, verb, adjective etc. I wanted to investigate whether including these would make a difference to the performance of the classifiers. So I created two test data sets, one that included pos tags and one that did not. Adding pos tags was done in between lemmatisaion and TF-IDF vectorisation.
<h5 align="center">POS Tags</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/part_of_speech.png" width=850> 
</p>


## Machine Learning Models
I created pipelines for 8 machine learning multiclassification models, with each model being trained on the data both with and without pos tags, for 16 pipelines in total. Each of these pipelines also includes hyperparameter optimisation using GridSearchCV. The models I used were  `Logistic Regression`, `Decision Tree`, `Naive Bayes`, `Support Vector Machine`,`Random Forest`,`K Nearest Neighbour`,`Stochastic Gradient Descent` and `Linear SVM`.

I then chose the 5 most accurate models and compared the impact that including pos tags had. As can be seen from the graph, including part of speech tags improved accuracy by over 10% for these models

<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/pos_accuracy_comparison.png" width=850> 
</p>

## Neural Networks
Additionally, I made 2 neural network classifiers. 

