import pandas as pd
import numpy as np
import nltk
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from flair.models import TextClassifier
from flair.data import Sentence
from sklearn.feature_extraction.text import TfidfVectorizer

stop_list = [
    'a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between',
    'both', 'but', 'by', 'can', 'could', 'd', 'did', 'do', 'does', 'doing', 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her',
    'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is',
    'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'more', 'most', 'my', 'myself',
    'need', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', 'she', 'should',
    'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 've', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
    'while', 'who', 'whom', 'why', 'will', 'with', 'won', 'would', 'y', 'you', 'your',
    'yours', 'yourself', 'yourselves'
]

def rating(num):
    if(num >=3):
        return 1
    else:
        return 0


def clean_text(text):
    if isinstance(text, str):
        text = ''.join([word for word in text if not any(c.isdigit() for c in word)])
        text = text.lower()
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        stop = [w for w in stop_list]
        text = [x for x in text if (x not in stop)]
    # remove empty tokens
        text = [t for t in text if len(t) > 0]
    # remove words with only one letter
        text = [t for t in text if len(t) > 1]
        text = [word for word in text if word]
        text = " ".join(text)
        return text
    elif isinstance(text, list):
        text = ''.join([word for word in text if not any(c.isdigit() for c in word)])
        text = text.lower()
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        stop = [w for w in stop_list]
        text = [x for x in text if (x not in stop)]
    # remove empty tokens
        text = [t for t in text if len(t) > 0]
    # remove words with only one letter
        text = [t for t in text if len(t) > 1]
        text = [word for word in text if word]
        text = " ".join(text)
        return text
    else:
      return ''
  
# def flair_prediction(x):
#     sentence = Sentence(x)
#     sia.predict(sentence)
#     label = sentence.labels[0]
#     if label.value == "POSITIVE":
#         return 1 
#     elif label.value == "NEGATIVE":
#         return -1
#     else:
#         return 0
    
    
def preprocess(df):
    # remove 'No Negative' or 'No Positive' from text
    df['Rating'] = df['Rating'].apply(rating)
    df["Reviews"] = df["Reviews"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", "")if isinstance(x, str) else x)
    df["review_clean"] = df["Reviews"].apply(lambda x: clean_text(x))
    df.dropna(subset=['review_clean'], inplace=True)
    df.dropna(subset=['Reviews'], inplace=True)
    return df

def sentiment_count(df):
    # [{'val': -1, 'freq': 5459}, {'val': 1, 'freq': 4541}]
    sentiment_counts = df['sentiment_score'].value_counts().reset_index()
    sentiment_counts = sentiment_counts.rename(columns={'index': 'val', 'sentiment_score': 'freq'})
    sentiment_counts_list = sentiment_counts.to_dict('records')
    return sentiment_counts_list

def get_top_positive_words(df):
    # [{'text': 'book', 'value': 837.6691894252903}, {'text': 'great', 'value': 663.3950063850567}]
    positive_df = df[df["sentiment_score"] == 1]
    vectorizer_positive = TfidfVectorizer(max_features=50)
    tfidf_positive = vectorizer_positive.fit_transform(positive_df["review_clean"])
    feature_names_positive = vectorizer_positive.get_feature_names_out()
    tfidf_scores_positive = tfidf_positive.toarray().sum(axis=0)
    word_scores_positive = list(zip(feature_names_positive, tfidf_scores_positive))
    word_scores_positive = sorted(word_scores_positive, key=lambda x: x[1], reverse=True)
    top_words_positive = word_scores_positive[:50]
    top_words_positive = [{'text': word, 'value': score} for word, score in top_words_positive]
    return top_words_positive

def get_top_negative_words(df):
    negative_df = df[df["sentiment_score"] == -1]
    vectorizer_negative = TfidfVectorizer(max_features=50)
    tfidf_negative = vectorizer_negative.fit_transform(negative_df["review_clean"])
    feature_names_negative = vectorizer_negative.get_feature_names_out()
    tfidf_scores_negative = tfidf_negative.toarray().sum(axis=0)
    word_scores_negative = list(zip(feature_names_negative, tfidf_scores_negative))
    word_scores_negative = sorted(word_scores_negative, key=lambda x: x[1], reverse=True)
    top_words_negative = word_scores_negative[:50]
    top_words_negative = [{'text': word, 'value': score} for word, score in top_words_negative]
    return top_words_negative
    
       
# df1 = pd.read_csv('output.csv')
# df=df1[['Rating','Title','Description']]
# df['Reviews'] = df['Title'] + ' . ' + df['Description']
# df = df.drop(['Title','Description'], axis=1)


# sia = TextClassifier.load('en-sentiment')

# df["sentiment_score"] = 0

# for i, row in df.iterrows():
#     sentiment_score = flair_prediction(row["Reviews"])
#     df.at[i, "sentiment_score"] = sentiment_score
    
