# -*- coding: utf-8 -*-
"""
@author: Erin S
"""
import pandas as pd
import numpy as np
import operator
from scipy import spatial
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Functions for content and collaborative based recommendation algorithm

def remove_dupe_neighbours(neighbours,restaurant_name_df):
    
    # Remove names that are very similar - use fuzzy comparison
    name_list = [restaurant_name_df['name'][restaurant_name_df.index == x[0]].values[0] for x in neighbours]
    name_list_no_dupes = list(process.dedupe(name_list,threshold = 90))
    elim =  list(set(name_list) - set(name_list_no_dupes))  # Names that were eliminated 

    index_list = [x[0] for x in neighbours]
    subset = restaurant_name_df.loc[index_list]
    subset = subset[~subset['name'].isin(elim)]
    
    # Remove exact duplicates
    exact_dupe_names = subset[subset['name'].duplicated(keep = False)]
    exact_dupe_names = exact_dupe_names.sort_values(by = ['name','stars'], ascending = True) # Drop match with fewest stars
    exact_dupe_names = exact_dupe_names.iloc[::2, :]   
    
    subset.drop(index = exact_dupe_names.index,inplace = True)
    neighbours_no_dupes = [neighbour_tuple for neighbour_tuple in neighbours if neighbour_tuple[0] in subset.index]
    
    return neighbours_no_dupes

def restaurant_similarity(restaurant_id1, restaurant_id2, restaurant_keywords_df, restaurant_categories_df):
    
    keywords_1 = restaurant_keywords_df.loc[restaurant_id1]
    keywords_2 = restaurant_keywords_df.loc[restaurant_id2]
    keyword_distance = spatial.distance.cosine(keywords_1, keywords_2)
    
    categories_1 = restaurant_categories_df.loc[restaurant_id1]
    categories_2 = restaurant_categories_df.loc[restaurant_id2] 
    category_distance = spatial.distance.cosine(categories_1, categories_2)
    
    return keyword_distance + category_distance

def get_restaurant_neighbours(orig_restaurant, restaurant_keywords_df, restaurant_categories_df, restaurant_name_df, K = 10):
  
    distances = []
    orig_name, orig_location, orig_stars = fetch_restaurant_description(orig_restaurant, restaurant_name_df)
        
    for current_index, restaurant in restaurant_name_df.iterrows():
            
        current_name = restaurant[0]
            
        if (current_index != orig_restaurant) and (orig_name not in current_name):
            restaurant_dist = restaurant_similarity(current_index, orig_restaurant, restaurant_keywords_df, restaurant_categories_df)
            distances.append((current_index, restaurant_dist)) # append tuple for distances
    
    distances.sort(key = operator.itemgetter(1))  # sort distances for nearest neighbours
    neighbors = []
    
    for x in range(K):
        neighbors.append(distances[x])
        
    return neighbors
    
def fetch_restaurant_description(restaurant_id, restaurant_name_df):
    
    name = restaurant_name_df.loc[restaurant_id]['name']
    location = restaurant_name_df.loc[restaurant_id]['address']
    stars = restaurant_name_df.loc[restaurant_id]['stars']
    
    return name, location, stars       
    
    
def print_recommendations(neighbour_tuple,restaurant_name_df):
    
    for restaurant in neighbour_tuple:
        name, location, stars = fetch_restaurant_description(restaurant[0], restaurant_name_df) 
        print("{}, {}, Stars: {} , distance: {} ".format(name,location,stars,restaurant[1]))


def allergen_scaling(business_id, df_contains,df_free_from ,df_dietary ,allergen_key_list = [], dietary_key_list = []):

# Compute penalty (positive/negative) to be added to the restaurant distance based off the relevant allergen indices 
    
    contains_penalty = []
    free_from_penalty = []
    allergy_accom_penalty = []
    dietary_penalty = []
    
    if allergen_key_list:
        
        for count,allergen in enumerate(allergen_key_list):

            # Contains Allergen: linear scaling with increased proportion of relevant allergen corresponding to larger penalty
            allergen_string = (str(allergen)+'_fraction').lower()
            contains_penalty.append(df_contains.loc[business_id][allergen_string])

            # Free From Allergen: linear scaling with dependence on average sentiment of relevant reviews
            allergen_sent_string = (str(allergen)+'_sentiment').lower()
            free_from_fraction = df_free_from.loc[business_id][allergen_string]
            free_from_sentiment = df_free_from.loc[business_id][allergen_sent_string]
            free_from_penalty.append(-1.1*free_from_fraction*free_from_sentiment)         # Positive sentiment - > negative penalty
     
    
        # Allergy Accomodations: linear scaling with dependence on average sentiment of relevant reviews
        # Always check allergy dietary accomodations regardless of specific allergen stated
        allergy_accom_fraction = df_dietary.loc[business_id]['allergy_fraction']
        allergy_accom_sentiment = df_dietary.loc[business_id]['allergy_sentiment']  
        allergy_accom_penalty.append(-1.1*allergy_accom_fraction*allergy_accom_sentiment) # Positive sentiment - > negative penalty

    if dietary_key_list:
       
        # Dietary Accomodations: linear scaling with dependence on average sentiment of relevant reviews
        for count, dietary in enumerate(dietary_key_list):
            dietary_string = (str(dietary)+'_fraction').lower()
            dietary_sent_string = (str(dietary)+'_sentiment').lower()
            dietary_fraction = df_dietary.loc[business_id][dietary_string]
            dietary_sentiment = df_dietary.loc[business_id][dietary_sent_string]
            dietary_penalty.append(-1.1*dietary_fraction*dietary_sentiment)              # Positive sentiment - > negative penalty
    
    penalty = np.array([contains_penalty, free_from_penalty, allergy_accom_penalty, dietary_penalty])
    final_penalty = np.nansum(np.sum(penalty, axis=0))
    
    return penalty, final_penalty
    
 
def re_rank_neighbours(neighbour_tuple, df_contains,df_free_from, df_dietary, allergen_key_list = [], dietary_key_list = []):
    
    new_neighbour_ranking = []
    
    for business_id, distance in neighbour_tuple:
        new_distance = 0
        penalty, final_penalty = allergen_scaling(business_id, allergen_key_list = allergen_key_list, dietary_key_list = dietary_key_list, df_contains = df_contains, df_free_from = df_free_from , df_dietary = df_dietary)
 
        if not np.isnan(final_penalty):
            new_distance = float(distance + final_penalty)
            
        else:
            new_distance = float(distance)
        new_neighbour_ranking.append((business_id,new_distance))
   
    new_neighbour_ranking.sort(key = operator.itemgetter(1))
    
    return new_neighbour_ranking   


def get_neighbour_user_ids(orig_user,user_item_df, K = 5):
    
    distances = []
        
    for user_id, current_user in user_item_df.iterrows():       
            
        user_distance = spatial.distance.cosine(orig_user, current_user)
        distances.append((user_id, user_distance)) 
    
    distances.sort(key = operator.itemgetter(1))  # sort distances for nearest neighbours
    neighbors = []
    
    for x in range(K):
        neighbors.append(distances[x])
        
    return neighbors

def get_restaurant_ids_from_neighbour_users(orig_user, neighbours, user_item_df):
    
    # Restaurants that are more highly rated that the user's average star rating
    recommend_ids = []
    already_rated = list(orig_user.columns[(orig_user > 0).all()])
    
    for neighbour in neighbours:
        
        user_ratings = user_item_df.loc[neighbour[0]]
        user_mean_rating = user_ratings[user_ratings!=0].mean()
        restaurants_to_rec = list(user_ratings[user_ratings>user_mean_rating].index)
        restaurants_to_rec = np.setdiff1d(restaurants_to_rec,already_rated)
        zipped = zip(restaurants_to_rec, [neighbour[1]]*len(restaurants_to_rec))
        recommend_ids.extend(zipped)
    
    return recommend_ids    

def scale_neighbour_distance(neighbour_tuple, to_add = 0):
    # Scale the distance of a neighbour list of tuples such that the first element has a distance of 0
    
    scaled_neighbours = []
    neighbour_tuple.sort(key = operator.itemgetter(1))
    min_distance = neighbour_tuple[0][1]
    
    for business_id, distance in neighbour_tuple:
        new_distance = distance - min_distance + to_add
        scaled_neighbours.append((business_id,new_distance))
        
    scaled_neighbours.sort(key = operator.itemgetter(1))
    
    return scaled_neighbours
