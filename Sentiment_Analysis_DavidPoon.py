#!dir /w
#!conda info --envs
#sys.executable

#!/usr/bin/env python
# coding: utf-8

#------------------------------------------------------------------------------
## MMA 865 - David Poon, Sid = 20198320
## Question 2 PART 1 - SENTIMENT ANLYSIS VIA ML-BASED APPROACH
#------------------------------------------------------------------------------

# Import packages
import pandas as pd
import numpy as np
import nltk

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))



#------------------------------------------------------------------------------
# Read in data
#------------------------------------------------------------------------------
import os
os.getcwd()

# change to desired filepath location

df = pd.read_csv("sentiment_train.csv")

#------------------------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS (EDA)
#------------------------------------------------------------------------------

df.shape
df.info()
df.head()
list(df)# list attribute names
dl = list(df) 
df.shape # instances, attributes
df.info() # data types
df.describe().transpose() # df attribute stats

# Check for nulls
dfNull = df.isnull().values.any().sum().sum()
print (df.isna())
print ('Checking Nulls: # of Obs')
print (dfNull)

# Check for duplicates
dfDup = df.duplicated().values.any().sum().sum()
print ('Checking Dups: # of Obs')
print (dfDup)


#------------------------------------------------------------------------------
# EDA: Plot and check for imbalance data
#------------------------------------------------------------------------------

#import matplotlib.pyplot as plt
#df2 = df
#ax = df2['Polarity'].value_counts(sort=False).plot(kind='barh')
#ax.set_xlabel("Number of Samples in training Set")
#ax.set_ylabel("Label")
#plt.show()

#----------------------------------------------------------------------------
# EDA: Get Language using stop words nltk
#----------------------------------------------------------------------------
import nltk
nltk.download("stopwords")
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

ENGLISH_STOPWORDS = set(stopwords.words('english'))
NON_ENGLISH_STOPWORDS = set(stopwords.words()) - ENGLISH_STOPWORDS
 
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}
 
def get_language(text):
    words = set(nltk.wordpunct_tokenize(text.lower()))
    return max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key = lambda x: x[1])[0]
 
 
def is_english(text):
    text = text.lower()
    words = set(nltk.wordpunct_tokenize(text))
    return len(words & ENGLISH_STOPWORDS) > len(words & NON_ENGLISH_STOPWORDS)


#lang_udf = udf(get_language,StringType())
#review_lang = df_lang.withColumn("reviewLang", lang_udf(df_lang.reviewText))
#review_lang1 = df_lang.withColumn("reviewLang", when(df_lang['reviewLength'] > 100, lang_udf(df_lang.concatText)).otherwise("english"))


print()
print ('#---------------------------------------------------')
print('DETECT LANGAUGES')
print ('#---------------------------------------------------')

# Check for different languages
df1=df
df1['language'] = df1['Sentence'].apply(lambda x: get_language(x))
df1.head(10)

lang_count=df1.groupby('language').count()
print(lang_count.head(10))
print()
print('Note: If length is too short, langauge detection can be flawed using stopwords method')

#df1.to_csv('lang.csv', index=False)


#------------------------------------------------------------------------------
# Vader Sentiment the dataset
#------------------------------------------------------------------------------

#!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df['Sentence']]
df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in df['Sentence']]
df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in df['Sentence']]
df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in df['Sentence']]
df

# Create sentiment feature threshold
def f(row):
    if row['compound'] > 0.10: #set threshold
        val = 1
    else:
        val = 0
    return val

df['Sentiment'] = df.apply(f, axis=1)
df
df.to_csv('scored_sentiment_sample.csv', index=False)


#------------------------------------------------------------------------------
# Split Train and Testing Data from the Sentiment_Train dataset
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X = df['Sentence']
y = df['Polarity']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=36)


#------------------------------------------------------------------------------
# Custom Functions for Preprocessing and Feature Engineering
#------------------------------------------------------------------------------
#!pip install textstat

print()
print ('#---------------------------------------------------')
print('DATA PREPROCESSING STEPS - Clean text & Feature Engineering')
print ('#---------------------------------------------------')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import unidecode
import textstat
import string  

lemmer = WordNetLemmatizer()

# Modified Prof Steve T's Simple preprocessor code base.
# Input is a single document, as a single string.
# Output should be a single document, as a single string.
def my_preprocess(doc):
    
    # Lowercase
    doc = doc.lower()
    
    # Replace URL with URL string
    doc = re.sub(r'http\S+', 'URL', doc)
    
    # Replace AT with AT string
    doc = re.sub(r'@', 'AT', doc)
    
    # Replace all numbers/digits with the string NUM
    doc = re.sub(r'\b\d+\b', 'NUM', doc)

    # Replace all #NAME ERROR with the string 'NAME ERROR'
    doc = re.sub(r'#NAME?\S+', 'NAME ERROR', doc)
     
    # Lemmatize each word.
    # doc = ' '.join([lemmer.lemmatize(w) for w in doc.split()])
  
    return doc


#----------------------------------------------------------------------------
# FEATURE ENGINEERING - Custom Functions to be used in Pipeline
#----------------------------------------------------------------------------

# These functions will calculate additional features on the document.
# They will be put into the Pipeline, called via the FunctionTransformer() function.
# Each one takes an entier corpus (as a list of documents), and should return
# an array of feature values (one for each document in the corpus).
# These functions can do anything they want; I've made most of them quick

# Apply vader sentiment analyzer compound, neg, neu, pos 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def lang(corpus):
    return np.array([lambda x: get_language(x) for doc in corpus]).reshape(-1,1)

def csentiment(corpus):
    return np.array([analyzer.polarity_scores(doc)['compound'] for doc in corpus]).reshape(-1,1)

def nsentiment(corpus):
    return np.array([analyzer.polarity_scores(doc)['neg'] for doc in corpus]).reshape(-1,1)

def neusentiment(corpus):
    return np.array([analyzer.polarity_scores(doc)['neu'] for doc in corpus]).reshape(-1,1)

def psentiment(corpus):
    return np.array([analyzer.polarity_scores(doc)['pos'] for doc in corpus]).reshape(-1,1)

def doc_length(corpus):
    return np.array([len(doc) for doc in corpus]).reshape(-1, 1)

def lexicon_count(corpus):
    return np.array([textstat.lexicon_count(doc) for doc in corpus]).reshape(-1, 1)

def _get_punc(doc):
    return len([a for a in doc if a in string.punctuation])

def punc_count(corpus):
    return np.array([_get_punc(doc) for doc in corpus]).reshape(-1, 1)

def _get_caps(doc):
    return sum([1 for a in doc if a.isupper()])

def capital_count(corpus):
    return np.array([_get_caps(doc) for doc in corpus]).reshape(-1, 1)

def num_exclamation_marks(corpus):
    return np.array([doc.count('!') for doc in corpus]).reshape(-1, 1)

def num_question_marks(corpus):
    return np.array([doc.count('?') for doc in corpus]).reshape(-1, 1)

def has_url(corpus):
    return np.array([bool(re.search("http", doc.lower())) for doc in corpus]).reshape(-1, 1)


#----------------------------------------------------------------------------
# CLASS WEIGHTS: To help handle class imbalance, calculate the class weights.
#----------------------------------------------------------------------------

import numpy as np
neg, pos = np.bincount(df['Polarity'])
total = neg + pos
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print()
print ('#---------------------------------------------------')
print ('# FIND CLASS WEIGHTS')
print ('#---------------------------------------------------')
print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

import matplotlib.pyplot as plt
ax = df['Polarity'].value_counts(sort=False).plot(kind='barh')
ax.set_xlabel("Number of Samples in training Set")
ax.set_ylabel("Label")

#----------------------------------------------------------------------------
# Construct Pipeline
#----------------------------------------------------------------------------

print()
print ('#---------------------------------------------------')
print ('# BUILDING PIPELINE')
print ('#---------------------------------------------------')

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import NMF
from sklearn.neural_network import MLPClassifier

# Need to preprocess the stopwords, because scikit learn's TfidfVectorizer
# removes stopwords _after_ preprocessing
stop_words = [my_preprocess(word) for word in stop_words.ENGLISH_STOP_WORDS]

# This vectorizer will be used to create the BOW features
vectorizer = TfidfVectorizer(preprocessor=my_preprocess, 
                             max_features = 1000, 
                             ngram_range=[1,4],
                             stop_words=None,
                             strip_accents="unicode", 
                             lowercase=False, max_df=0.25, min_df=0.001, use_idf=True)

# This vectorizer will be used to preprocess the text before topic modeling.
# (I _could_ use the same vectorizer as above- but why limit myself?)
vectorizer2 = TfidfVectorizer(preprocessor=my_preprocess, 
                             max_features = 1000, 
                             ngram_range=[1,2],
                             stop_words=None,
                             strip_accents="unicode", 
                             lowercase=False, max_df=0.25, min_df=0.001, use_idf=True)

#----------------------------------------------------------------------------
## Algos
#----------------------------------------------------------------------------
nmf = NMF(n_components=25, random_state=1, init='nndsvda', solver='mu', alpha=.1, l1_ratio=.5)
rf = RandomForestClassifier(criterion='entropy', random_state=223)
mlp = MLPClassifier(random_state=42, verbose=2, max_iter=200)



feature_processing =  FeatureUnion([ 
    ('bow', Pipeline([('cv', vectorizer), ])),
    ('topics', Pipeline([('cv', vectorizer2), ('nmf', nmf),])),
    #('language', FunctionTransformer(lang, validate=False)),
    ('sentiment_compound', FunctionTransformer(csentiment, validate=False)),
    ('sentiment_neg', FunctionTransformer(nsentiment, validate=False)),
    ('sentiment_neu', FunctionTransformer(neusentiment, validate=False)),
    ('sentiment_pos', FunctionTransformer(psentiment, validate=False)),
    ('length', FunctionTransformer(doc_length, validate=False)),
    ('words', FunctionTransformer(lexicon_count, validate=False)),
    ('punc_count', FunctionTransformer(punc_count, validate=False)),
    ('capital_count', FunctionTransformer(capital_count, validate=False)),  
    ('num_exclamation_marks', FunctionTransformer(num_exclamation_marks, validate=False)),  
    ('num_question_marks', FunctionTransformer(num_question_marks, validate=False)),   
    ('has_url', FunctionTransformer(has_url, validate=False))
])

steps = [('features', feature_processing)]

pipe = Pipeline([('features', feature_processing), ('clf', mlp)])

param_grid = {}

# You - yes you! Manually choose which classifier run you'd like to try.
# In future I'd like to automate this so that both are tried; but for this simple
# Kaggle competition, I'm keeping it simple. You can set this to either:
#
# "RF" - Random Forest
# "MLP" - NN

which_clf = "MLP"

if which_clf == "RF":

    steps.append(('clf', rf))

    # I already ran a 4-hour extensive grid; this is not the full set. BTW, the best hyperarms I found are:
    # Best parameter (CV scy_train0.988):
    # {'clf__class_weight': None, 
    # 'clf__n_estimators': 500, 
    # 'features__bow__cv__max_features': 500, 
    # 'features__bow__cv__preprocessor': None, 
    # 'features__bow__cv__use_idf': False, 
    # 'features__topics__cv__stop_words': None, 
    # 'features__topics__nmf__n_components': 300}
    param_grid = {
        'features__bow__cv__preprocessor': [None, my_preprocess],
        'features__bow__cv__max_features': [200, 500, 1000],
        'features__bow__cv__use_idf': [False],
        'features__topics__cv__stop_words': [None],
        'features__topics__nmf__n_components': [25, 75],
        'clf__n_estimators': [100, 500],
        'clf__class_weight': [None],
    }
    
elif which_clf == "MLP":
    
    steps.append(('clf', mlp))

    # I already ran a 4-hour extensive grid; this is not the full set. BTW, the best hyperarms I found are:
    # Best parameter (CV scy_train0.991): 
    # {'clf__hidden_layer_sizes': (25, 25, 25), 
    # 'features__bow__cv__max_features': 3000, 
    # 'features__bow__cv__min_df': 0, 
    # 'features__bow__cv__preprocessor': <function my_preprocess at 0x0000024801E161E0>, 
    # 'features__bow__cv__use_idf': False, 
    # 'features__topics__nmf__n_components': 300}
    param_grid = {
        'features__bow__cv__preprocessor': [my_preprocess],
        'features__bow__cv__max_features': [3000],
        'features__bow__cv__min_df': [0],
        'features__bow__cv__use_idf': [False],
        'features__topics__nmf__n_components': [300],
        'clf__hidden_layer_sizes': [(25, 25, 25)],
    }

pipe = Pipeline(steps)

search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=3, scoring='f1_micro', return_train_score=True, verbose=2)


#----------------------------------------------------------------------------
# Fit Model
#----------------------------------------------------------------------------
print()
print ('#---------------------------------------------------')
print ('# Fitting Model --> Its show time, baby')
print ('#---------------------------------------------------')

search = search.fit(X_train, y_train)

print("Best parameter (CV scy_train%0.3f):" % search.best_score_)
print(search.best_params_)


# Print out the results of hyperparmater tuning

def cv_results_to_df(cv_results):
    results = pd.DataFrame(list(cv_results['params']))
    results['mean_fit_time'] = cv_results['mean_fit_time']
    results['mean_score_time'] = cv_results['mean_score_time']
    results['mean_train_score'] = cv_results['mean_train_score']
    results['std_train_score'] = cv_results['std_train_score']
    results['mean_test_score'] = cv_results['mean_test_score']
    results['std_test_score'] = cv_results['std_test_score']
    results['rank_test_score'] = cv_results['rank_test_score']

    results = results.sort_values(['mean_test_score'], ascending=False)
    return results

results = cv_results_to_df(search.cv_results_)
results
results.to_csv('results2.csv', index=False)


#----------------------------------------------------------------------------
# Estimate Model Performance on Val Data
#----------------------------------------------------------------------------
# Because we are using a pipeline and a GridSearchCV, things are a bit complicated.
# I want to get references to the objects from the pipeline with the *best* hyperparameter settings,
# so that I can explore those objects (later). 
# The code below is a bit ugly, but after reading throught the docs of Pipeline, 
# I believe this is the only way to do it.
print()
print ('#---------------------------------------------------')
print ('# NN MLP on TRAINING DATA')
print ('# Model results on sentiment "TRAINING" data')
print ('#---------------------------------------------------')


# The pipeline with the best performance
pipeline = search.best_estimator_

# Get the feature processing pipeline, so I can use it later
feature_processing_obj = pipeline.named_steps['features']

# Find the vectorizer objects, the NMF objects, and the classifier objects
pipevect= dict(pipeline.named_steps['features'].transformer_list)
vectorizer_obj = pipevect.get('bow').named_steps['cv']
vectorizer_obj2 = pipevect.get('topics').named_steps['cv']
nmf_obj = pipevect.get('topics').named_steps['nmf']
clf_obj = pipeline.named_steps['clf']

# Sanity check - what was vocabSize set to? Should match the output here.
len(vectorizer_obj.get_feature_names())


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score, accuracy_score,  matthews_corrcoef, roc_auc_score

features_val = feature_processing_obj.transform(X_val).todense()

pred_val = search.predict(X_val)

print("\nMCC Score = {:.5f}".format(matthews_corrcoef(y_val, pred_val)))
print("\nroc_AUC Score = {:.5f}".format(roc_auc_score(y_val, pred_val)))
print("\n")

print("Confusion matrix:")
print(confusion_matrix(y_val, pred_val))

print("\nF1 Score = {:.5f}".format(f1_score(y_val, pred_val, average='micro')))

print("\nClassification Report:")
print(classification_report(y_val, pred_val))


#----------------------------------------------------------------------------
# Estimate Performance on Test/Kaggle Data
#----------------------------------------------------------------------------
test_df = pd.read_csv('sentiment_test.csv')
test_df.shape
test_df.head(5)

test_df['ID'] = test_df.index
test_df

#----------------------------------------------------------------------------
# VADER SENTIMENT ON TEST DATA
#----------------------------------------------------------------------------

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

test_df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in test_df['Sentence']]
test_df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in test_df['Sentence']]
test_df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in test_df['Sentence']]
test_df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in test_df['Sentence']]

def f(row):
    if row['compound'] > 0.10: #set threshold
        val = 1
    else:
        val = 0
    return val

test_df['Sentiment'] = test_df.apply(f, axis=1)
test_df



#----------------------------------------------------------------------------
# MODEL ON TEST DATA
#----------------------------------------------------------------------------
print()
print ('#---------------------------------------------------')
print ('# NN MLP on TEST SET')
print ('# Model results on sentiment "TEST" data')
print ('#---------------------------------------------------')

features_test = feature_processing_obj.transform(test_df['Sentence']).todense()
pred_test = search.predict(test_df['Sentence'])

# Output the predictions to a file to upload to Kaggle.
# Uncomment to actually create the file
my_submission = pd.DataFrame({'ID': test_df.ID, 
                              'Sentence': test_df.Sentence, 
                              'Polarity': test_df.Polarity, 
                              'predicted': pred_test})

# Need to convert into a series
s = pd.Series(pred_test)
my_submission.to_csv('out.csv', index=False)

y_test = test_df['Polarity']

print("\nMCC Score = {:.5f}".format(matthews_corrcoef(y_test, pred_test)))
print("\nroc_AUC Score = {:.5f}".format(roc_auc_score(y_val, pred_val)))
print("\n")

print("Confusion matrix:")
print(confusion_matrix(y_test, pred_test))

print("\nF1 Score = {:.5f}".format(f1_score(y_test, pred_test, average="micro")))

print("\nClassification Report:")
print(classification_report(y_test, pred_test))


#----------------------------------------------------------------------------
# MLP Hyper Parameters
#----------------------------------------------------------------------------
#Best parameter (CV scy_train0.782):
#{'clf__hidden_layer_sizes': (50, 50), 'features__bow__cv__max_features': 3000, 'features__bow__cv__min_df': 0, #'features__bow__cv__preprocessor': <function my_preprocess at 0x0000029F4A09DCA8>, 'features__bow__cv__use_idf': False, #'features__topics__nmf__n_components': 300}


print()
print ('#---------------------------------------------------')
print ('# END OF PROGRAM ')
print ('#---------------------------------------------------')
print("THATS A WRAP - PGM END")



# ## Print Feature Importances
# 
# Note: this section will only work with models that have `.feature_importances_`, such as RF and DT.

# In[21]:


topic_feature_names = ["topic {}".format(i) for i in range(nmf_obj.n_components_)]

stat_feature_names = [t[0] for t in pipeline.named_steps['features'].transformer_list if t[0] not in ['topics', 'bow']]

feature_names = vectorizer_obj.get_feature_names() + topic_feature_names + stat_feature_names
len(feature_names)

feature_importances = None
if hasattr(clf_obj, 'feature_importances_'):
    feature_importances = clf_obj.feature_importances_


# In[22]:


features_train = feature_processing_obj.transform(X_train).todense()

if feature_importances is None:
    print("No Feature importances! Skipping.")
else:
    N = features_train.shape[1]

    ssum = np.zeros(N)
    avg = np.zeros(N)
    avg_spam = np.zeros(N)
    avg_ham = np.zeros(N)
    for i in range(N):
        ssum[i] = sum(features_train[:, i]).reshape(-1, 1)
        avg[i] = np.mean(features_train[:, i]).reshape(-1, 1)
        avg_spam[i] = np.mean(features_train[y_train==1, i]).reshape(-1, 1)
        avg_ham[i] = np.mean(features_train[y_train==0, i]).reshape(-1, 1)

    rf = search.best_estimator_
    imp = pd.DataFrame(data={'feature': feature_names, 'imp': feature_importances, 'sum': ssum, 'avg': avg, 'avg_ham': avg_ham, 'avg_spam': avg_spam})
    imp = imp.sort_values(by='imp', ascending=False)
    imp.head(20)
    imp.tail(10)
    imp.to_csv('importances.csv', index=False)


# # Further explanation on Val Data
# 
# This cool package will explain all the predictions of a tree-based model. I'll have it explain all predictions that were incorrect, to see what is going on (and hopefully inform some additional feature engineering or cleaning steps).
# 
# Note: this only works on tree-based models, like RF. This cell will crash when using, e.g., MLPClassifier

# In[23]:


#!pip install treeinterpreter


# In[24]:


if feature_importances is None:
    print("No Feature importances! Skipping.")
else:

    from treeinterpreter import treeinterpreter as ti

    prediction, bias, contributions = ti.predict(clf_obj, features_val)

    for i in range(len(features_val)):
        if y_val.iloc[i] == pred_val[i]:
            continue
        print("Instance {}".format(i))
        X_val.iloc[i]
        print("Bias (trainset mean) {}".format(bias[i]))
        print("Truth {}".format(y_val.iloc[i]))
        print("Prediction {}".format(prediction[i, :]))
        print("Feature contributions:")
        con = pd.DataFrame(data={'feature': feature_names, 
                                 'value': features_val[i].A1,
                                 'legit contr': contributions[i][:, 0],
                                 'spam contr': contributions[i][:, 1],
                                 'abs contr': abs(contributions[i][:, 1])})

        con = con.sort_values(by="abs contr", ascending=False)
        con['spam cumulative'] = con['spam contr'].cumsum() + bias[i][1]
        con.head(30)
        print("-"*20) 


# # Further exploration on Test/Kaggle Data
# 
# Note: this only works on tree-based models, like RF. This cell will crash when using, e.g., MLPClassifier

# In[25]:


if  feature_importances is None:
    print("No Feature importances! Skipping.")
else:

    from treeinterpreter import treeinterpreter as ti

    prediction, bias, contributions = ti.predict(clf_obj, features_test)

    for i in range(len(features_test)):
        if y_test[i] == pred_test[i]:
            continue
        print("Instance {}".format(i))
        test_df.iloc[i, :].Sentence
        print("Bias (trainset mean) {}".format(bias[i]))
        print("Truth {}".format(y_test[i]))
        print("Prediction {}".format(prediction[i, :]))
        print("Feature contributions:")
        con = pd.DataFrame(data={'feature': feature_names,
                                 'value': features_test[i].A1,
                                 'legit contr': contributions[i][:, 0],
                                 'spam contr': contributions[i][:, 1],
                                 'abs contr': abs(contributions[i][:, 1])})
        con = con.sort_values(by="abs contr", ascending=False)
        con['spam cumulative'] = con['spam contr'].cumsum() + bias[i][1]
        con.head(30)
        print("-"*20) 
