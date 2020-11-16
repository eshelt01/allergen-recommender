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

# Content-based neighbours from given restaurants
given_restaurant_id1 = restaurant_name_df[restaurant_name_df['name']==given_restaurant_name_1].index[0]
given_restaurant_id2 = restaurant_name_df[restaurant_name_df['name']==given_restaurant_name_2].index[0]

content_neighbours_1 = rec.get_restaurant_neighbours(given_restaurant_id1, restaurant_keywords_df, restaurant_categories_df, restaurant_name_df, K = 50)
content_neighbours_2 = rec.get_restaurant_neighbours(given_restaurant_id2, restaurant_keywords_df, restaurant_categories_df, restaurant_name_df, K = 50)

# Based off of user's input restaurants, create mock user to compare to existing users
mock_user = pd.DataFrame(0.0, columns=user_item_df.columns, index = ['mock_user'])
mock_user[[given_restaurant_id1,given_restaurant_id2]] = 5.0

# Neighbours to mock user from user-item filtering
neighbours_to_mock_user = rec.get_neighbour_user_ids(mock_user, user_item_df, K = 10)
user_neighbours = rec.get_restaurant_ids_from_neighbour_users(mock_user, neighbours_to_mock_user, user_item_df = user_item_df)

#Combine all neighbours and remove duplicates and close name matches (for chain restaurants)
combined_neighbours = rec.scale_neighbour_distance(content_neighbours_1) + rec.scale_neighbour_distance(content_neighbours_2) + rec.scale_neighbour_distance(user_neighbours,to_add = 0.05)
combined_neighbours = rec.remove_dupe_neighbours(combined_neighbours,restaurant_name_df)

# Re-rank neighbours based off of user's input allergies and dietary accomodation requests
allergy_adjusted_neighbours = rec.re_rank_neighbours(combined_neighbours, df_contains,df_free_from, df_dietary, given_allergen_key_list, given_dietary_key_list)

# Print top 10 recommendations
rec.print_recommendations(allergy_adjusted_neighbours[:10],restaurant_name_df)