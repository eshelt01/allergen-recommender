# -*- coding: utf-8 -*-
"""
@author: Erin S
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer


def unique_business_ids(review_df, time_cutoff = '2000-01-01'):    
    
    unique_business_id = review_df['business_id'][review_df['date']>time_cutoff].unique()
    return unique_business_id            


def sort_matrix(coo_matrix):
    # Reformat sparse COO matrix
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse = True)


def extract_top_n_keywords(feature_names, sorted_items, top_n = 10):
    
    sorted_items = sorted_items[:top_n] # top n keywords
    keywords = []    

    for index, score in sorted_items:        
        keywords.append(feature_names[index])
    
    return keywords


def get_restaurant_keywords(business_id,review_df, tfidf_Transformer, Count_Vectorizer, feature_names, top_n = 10):
   
    review_text = review_df['text'][review_df['business_id']== business_id].to_list()
    review_text = ' '.join(review_text)  # Join all from this restaurant into reviews into one document
    
    tf_idf_vector = tfidf_Transformer.transform(Count_Vectorizer.transform([review_text]))
    sorted_items = sort_matrix(tf_idf_vector.tocoo())
    keywords = extract_top_n_keywords(feature_names,sorted_items,top_n) # top 20 keywords for each restaurant
    
    return keywords

# Import reviews and corpus vocabulary 
review_df = pd.read_pickle('./data/restaurant_reviews.pkl')   # Reviews
vocab_df = pd.read_pickle('./data/corpus_vocabulary.pkl')     # Corpus vocabulary
text = vocab_df['text'].to_list()

# Create count vectorizer
stop_nltk = set(stopwords.words("english"))
food_words = ['place','food','dish','good','us','menu','order','drink','eat','would','could','restaurant','store','like','even','came','nice','great','time','love','always','one','went','get','got','really','also','another','ordered','go','try','well','definitely','come','made','back','dishes','little','small','thing','less','away','put','usually','lot','served']
stop_words = set(stop_nltk.union(food_words))

CV = CountVectorizer(max_df = 0.85, min_df = 0.01, stop_words =stop_words)
cv_vector = CV.fit_transform(text)

# Create TFIDF transformer
tfidf_Transformer = TfidfTransformer(smooth_idf = True, use_idf = True)
tfidf_Transformer.fit(cv_vector)
feature_names = CV.get_feature_names()

# Creating dataframe for tfidf for each restaurant
unique_restaurant_ids = unique_business_ids(review_df = review_df)   
restaurant_tfidf = pd.DataFrame(columns = ['keywords'])

for index, restaurant_id in enumerate(unique_restaurant_ids):

    top_keywords = get_restaurant_keywords(restaurant_id,review_df = review_df,tfidf_Transformer = tfidf_Transformer, Count_Vectorizer = CV,feature_names = feature_names, top_n = 20)
    restaurant_tfidf.loc[restaurant_id] = [top_keywords]

# Organize dataframe with each unique keyword as column and restaurant if as index    
mlb = MultiLabelBinarizer()
restaurant_tfidf_processed = pd.DataFrame(mlb.fit_transform(restaurant_tfidf['keywords']), columns = mlb.classes_, index = restaurant_tfidf.index)

# Filter keywords by restaurant frequency, only keep most frequent and relevant keywords
keyword_freq = pd.DataFrame(columns = ['frequency'])

for column in restaurant_tfidf_processed:
    keyword_freq.loc[column] = np.mean(restaurant_tfidf_processed[column])


words_to_keep = keyword_freq[keyword_freq['frequency']>0.03].index 
restaurant_tfidf_processed = restaurant_tfidf_processed[words_to_keep]

#restaurant_tfidf_processed.to_pickle('./restaurant_TFIDF_keywords_processed.pkl')







