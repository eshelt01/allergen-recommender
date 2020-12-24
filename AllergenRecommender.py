# -*- coding: utf-8 -*-
"""

@author: Erin S
"""

import pandas as pd
import hybrid_recommend as rec

#                             Recommender
#----------------------------------------------------------------------------#

#Import dataframes
review_df = pd.read_pickle('./data/restaurant_reviews.pkl')
restaurant_name_df = pd.read_pickle('./data/restaurant_name_info.pkl')
user_item_df = pd.read_pickle('./data/user_item_review_stars.pkl')

restaurant_keywords_df = pd.read_pickle('./data/restaurant_TFIDF_keywords_processed.pkl')
restaurant_categories_df = pd.read_pickle('./data/restaurant_categories_cleaned.pkl')

df_contains = pd.read_pickle('./data/df_allergen_contains.pkl')
df_free_from = pd.read_pickle('./data/df_allergen_free_from.pkl')
df_dietary = pd.read_pickle('./data/df_dietary_accomodations.pkl')

# Reference allergen and dietary key list options
allergen_key_master_list = ['milk','egg','fish','crustacean','tree_nut','peanut','wheat','soy']
dietary_key_master_list = ['vegan', 'vegetarian', 'celiac']


#User input -> two restaurants that they already enjoy
given_restaurant_name_1 = 'STACK'
given_restaurant_name_2 = 'Pizzeria Libretto'

#User input -> allergies and dietary accomodations
given_allergen_key_list = ['peanut']
given_dietary_key_list = ['vegetarian']

#User input -> two restaurants that they already enjoy
restaurant_names = ['STACK','Pizzeria Libretto']

#User input -> allergies and dietary accomodations
given_allergen_key_list = ['peanut']
given_dietary_key_list = ['vegetarian']

# Content-based neighbours from given restaurants

neighbours = rec.recommend(restaurant_names, given_allergen_key_list, given_dietary_key_list, restaurant_name_df, restaurant_keywords_df, restaurant_categories_df, user_item_df, df_contains,df_free_from, df_dietary)

# Print top 20 recommendations
rec.print_recommendations(neighbours[:20],restaurant_name_df)
