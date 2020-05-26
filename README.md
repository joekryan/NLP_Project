# NLP Classification

## File Descriptions
- EDA_and_preprocessing.ipynb: notebook for obtaining and cleaning data, and processing the text data into a TF-IDF vector
- machine_learning_classification_models.ipynb: notebook where machine learning models are tuned, trained and evaluated
- NeuralNets.ipynb: notebook where neural network classifier models are built, trained and evaluated
- best_model_and_evaluation.ipynb: notebook where the best model is chosen and evaluated, and ideas for further work are explored
- Data: folder that contains .csv files for train, test and validation datasets, both with and without part of speech tags
- Images: folder that contains visualisations  

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
Additionally, I made 2 neural network classifiers. The first neural net took as its input the TF-IDF matrix. This was fairly simple in its structure but had a high number of input dimensions due to using the TF-IDF matrix. As with the machine learning models, I did this both with and without pos tags. Again, pos tags did improve the performance but only by 1%, not by 10% as before. The best performing neural network had an accuracy of c.87%.

<h5 align="center">Architecture for Neural Network that uses TF-IDF</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/initial_nn.png" width=850> 
</p>

The second neural network had a more complex architecture, using a precomputed GloVe vector embedding layer. I chose a GloVe vector that had been trained on twitter, as microblogging posts seemed the most similar to the newsgroup posts. This second neural network, despite being more complex was slightly less accurate than the first, with an accuracy of c.85%.

<h5 align="center">Architecture for Neural Network that uses GloVe Embedded Vector</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/cnn_model.png" width=850> 
</p>

## Choosing the Best Model 
When choosing the best model, I considered both accuracy and ease of implementation. My first thought was to use the best machine learning classifiers to create a stacking classifier. However, comparing the confusion matrices (using the validation dataset) for the best 5 machine learning classifiers, it can be seen that they all have the same strengths and weaknesses. The best neural network classifier also shared similar characteristics. Stacking classifiers with the same strengths and weaknesses will have negligible effect on accuracy for a great increase in computational complexity.

Notably, the classifiers all have the most trouble in accuractely classifying the talk.religion.misc category, often miscateogrising it as religion.talk.christian or alt.atheism. Additionally, there is significant miscategorisation amongst the 5 comp.* classes.

<h5 align="center">Confusion Matrices for the 5 Best Machine Learning Classifiers</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/ml_collage.png" width=850> 
</p>

In this case, the choice for choosing a best model was between the best machine learning classifier, Linear SVM, with an accuracy of c.85%, and the best neural network classifier that had an accuracy of c.87%. I chose Linear SVM as the best model, as the difference in accuracy between the two is small, but the difference in computationl complexity and time required to train the model and make predictions is large.

## Best Model Results

accuracy:  0.8434
f1_score: 0.8434
precision_score: 0.8434
recall_score: 0.8434

<h5 align="center">Best Model (Linear SVC) Confusion Matrix</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/best_model_confusion_matrix.png" width=850> 
</p>

<h5 align="center">Best Model (Linear SVC) Roc Curve</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/best_model_roc_curve.png" width=850> 
</p>

<h5 align="center">Best Model (Linear SVC) Classificaiton Report</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/best_model_classification_report.png" width=850> 
</p>

## Future Work
The 20 newsgroup classes consist of 7 superclasses, alt, comp, misc, rec, sci, soc and talk. Due to the confusion between classes and within classes that the classifiers all had, it would be a good approach sequentially classify the data, to first separate the classes into the 7 superclasses and then into the final classes within each superclass.

To test if this would be a good approach, I took the best model and used it to classify the 7 superclasses, and just within the comp.* superclass. As I expected, the model was much more accurate at prediction just within the comp.* superclass than it was when classifying across all classes. The classifier also had a >90% accuracy for predicting the superclasses.

<h5 align="center">Superclass Confusion Matrix</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/superclass_confusion_matrix.png" width=850> 
</p>

<h5 align="center">Comp.* Confusion Matrix</h5>
<p align="center">
  <img src="https://github.com/joekryan/nlp_classification/blob/master/Images/comp_confusion_matrix.png" width=850> 
</p>

Given that this model had not been specifically trained or optimised to do these tasks, it seems that this would be a way to get a more accurate model. However, given that there would need to be separate train/test splits at each stage as well as separate models and optimisations, this would lead to a much more complex solution. Whether this would be practical or not would depend on the situation, i.e. would an extra 10% accuracy be worth a great increase in complexity. 






