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

class hybridRecommend():
    
    def __init__(self, allergen_key_list = [], dietary_key_list = []):
       
        self.restaurant_name_df = pd.read_pickle('C:/Users/Ed/Documents/allergen-recommender/data/restaurant_name_info.pkl')
        self.user_item_df = pd.read_pickle('C:/Users/Ed/Documents/allergen-recommender/data/user_item_review_stars.pkl')
        self.restaurant_keywords_df = pd.read_pickle('C:/Users/Ed/Documents/allergen-recommender/data/restaurant_TFIDF_keywords_processed.pkl')
        self.restaurant_categories_df = pd.read_pickle('C:/Users/Ed/Documents/allergen-recommender/data/restaurant_categories_cleaned.pkl')
        self.df_contains = pd.read_pickle('C:/Users/Ed/Documents/allergen-recommender/data/df_allergen_contains.pkl')
        self.df_free_from = pd.read_pickle('C:/Users/Ed/Documents/allergen-recommender/data/df_allergen_free_from.pkl')
        self.df_dietary = pd.read_pickle('C:/Users/Ed/Documents/allergen-recommender/data/df_dietary_accomodations.pkl')

        self.allergen_key_list = allergen_key_list 
        self.dietary_key_list = dietary_key_list
        #self.neighbours = []
        
        return
    

    # Functions for content and collaborative based recommendation algorithm
    
    def remove_dupe_neighbours(self, neighbours):
        """ Removes duplicates from list of restaurant neighbours """
        
        # Remove names that are very similar - use fuzzy comparison
        name_list = [self.restaurant_name_df['name'][self.restaurant_name_df.index == x[0]].values[0] for x in neighbours]
        name_list_no_dupes = list(process.dedupe(name_list,threshold = 90))
        elim =  list(set(name_list) - set(name_list_no_dupes))  # Names that were eliminated 
    
        index_list = [x[0] for x in neighbours]
        subset = self.restaurant_name_df.loc[index_list]
        subset = subset[~subset['name'].isin(elim)]
        
        # Remove exact duplicates
        exact_dupe_names = subset[subset['name'].duplicated(keep = False)]
        exact_dupe_names = exact_dupe_names.sort_values(by = ['name','stars'], ascending = True) # Drop match with fewest stars
        exact_dupe_names = exact_dupe_names.iloc[::2, :]   
        
        subset.drop(index = exact_dupe_names.index,inplace = True)
        neighbours_no_dupes = [neighbour_tuple for neighbour_tuple in neighbours if neighbour_tuple[0] in subset.index]
        
        return neighbours_no_dupes
    
    def restaurant_similarity(self,restaurant_id1, restaurant_id2):
        """ Calculate cosine similarity between restaurants """
        
        keywords_1 = self.restaurant_keywords_df.loc[restaurant_id1]
        keywords_2 = self.restaurant_keywords_df.loc[restaurant_id2]
        keyword_distance = spatial.distance.cosine(keywords_1, keywords_2)
        
        categories_1 = self.restaurant_categories_df.loc[restaurant_id1]
        categories_2 = self.restaurant_categories_df.loc[restaurant_id2] 
        category_distance = spatial.distance.cosine(categories_1, categories_2)
        
        return keyword_distance + category_distance
    
    def get_restaurant_neighbours(self, orig_restaurant, K = 10):
        """ Find all K neighbours to given restaurant id """
        
        distances = []
        orig_name, orig_location, orig_stars = self.fetch_restaurant_description(orig_restaurant)
            
        for current_index, restaurant in self.restaurant_name_df.iterrows():
                
            current_name = restaurant[0]
                
            if (current_index != orig_restaurant) and (orig_name not in current_name):
                restaurant_dist = self.restaurant_similarity(current_index, orig_restaurant)
                # append tuple for distances
                distances.append((current_index, restaurant_dist)) 
        
        # sort distances for nearest neighbours
        distances.sort(key = operator.itemgetter(1))  
        neighbors = []
        
        for x in range(K):
            neighbors.append(distances[x])
            
        return neighbors
        
    def fetch_restaurant_description(self,restaurant_id):
        """ Returns name, location and stars of given restaurant """
        
        name = self.restaurant_name_df.loc[restaurant_id]['name']
        location = self.restaurant_name_df.loc[restaurant_id]['address']
        stars = self.restaurant_name_df.loc[restaurant_id]['stars']
        
        return name, location, stars       
        
        
    def print_recommendations(self,neighbour_tuple, number = 10):
        """ Prints stars, distance, name and location of given restaurant """
        
        for restaurant in neighbour_tuple[:number]:
            name, location, stars = self.fetch_restaurant_description(restaurant[0]) 
            print("{}, {}, Stars: {} , distance: {} ".format(name,location,stars,restaurant[1]))
    
    
    def allergen_scaling(self, business_id):
        """ Computes the custom penalty for restaurants in neighbour list. """ 
    
    # Compute penalty (positive/negative) to be added to the restaurant distance based off the relevant allergen indices  
    # Positive sentiment - > negative penalty       
        contains_penalty = []
        free_from_penalty = []
        allergy_accom_penalty = []
        dietary_penalty = []
        
        if self.allergen_key_list:
            
            for count,allergen in enumerate(self.allergen_key_list):
    
                # Contains Allergen: linear scaling with increased proportion of relevant allergen corresponding to larger penalty
                allergen_string = (str(allergen)+'_fraction').lower()
                contains_penalty.append(self.df_contains.loc[business_id][allergen_string])
    
                # Free From Allergen: linear scaling with dependence on average sentiment of relevant reviews
                allergen_sent_string = (str(allergen)+'_sentiment').lower()
                free_from_fraction = self.df_free_from.loc[business_id][allergen_string]
                free_from_sentiment = self.df_free_from.loc[business_id][allergen_sent_string]
                free_from_penalty.append(-1.1*free_from_fraction*free_from_sentiment)        
         
        
            # Allergy Accomodations: linear scaling with dependence on average sentiment of relevant reviews
            # Always check allergy dietary accomodations regardless of specific allergen stated
            allergy_accom_fraction = self.df_dietary.loc[business_id]['allergy_fraction']
            allergy_accom_sentiment = self.df_dietary.loc[business_id]['allergy_sentiment']  
            allergy_accom_penalty.append(-1.1*allergy_accom_fraction*allergy_accom_sentiment) 
    
        if self.dietary_key_list:
           
            # Dietary Accomodations: linear scaling with dependence on average sentiment of relevant reviews
            for count, dietary in enumerate(self.dietary_key_list):
                dietary_string = (str(dietary)+'_fraction').lower()
                dietary_sent_string = (str(dietary)+'_sentiment').lower()
                dietary_fraction = self.df_dietary.loc[business_id][dietary_string]
                dietary_sentiment = self.df_dietary.loc[business_id][dietary_sent_string]
                dietary_penalty.append(-1.1*dietary_fraction*dietary_sentiment)              
        
        penalty = np.array([contains_penalty, free_from_penalty, allergy_accom_penalty, dietary_penalty])
        final_penalty = np.nansum(np.sum(penalty, axis=0))
        
        return penalty, final_penalty
        
     
    def re_rank_neighbours(self, neighbour_tuple):
        """ Re-rank restaurant neighbours by custom allrgen penalty """
       
        new_neighbour_ranking = []
        
        for business_id, distance in neighbour_tuple:
            new_distance = 0
            penalty, final_penalty = self.allergen_scaling(business_id)
     
            if not np.isnan(final_penalty):
                new_distance = float(distance + final_penalty)
                
            else:
                new_distance = float(distance)
            new_neighbour_ranking.append((business_id,new_distance))
       
        new_neighbour_ranking.sort(key = operator.itemgetter(1))
        
        return new_neighbour_ranking   
    
    
    def get_neighbour_user_ids(self, orig_user, K = 5):
        """ Find neighbours to given user """
        
        distances = []
            
        for user_id, current_user in self.user_item_df.iterrows():       
                
            user_distance = spatial.distance.cosine(orig_user, current_user)
            distances.append((user_id, user_distance)) 
        
        distances.sort(key = operator.itemgetter(1))  # sort distances for nearest neighbours
        neighbors = []
        
        for x in range(K):
            neighbors.append(distances[x])
            
        return neighbors
    
    def get_restaurant_ids_from_neighbour_users(self, orig_user, neighbours):
        """ Find restaurants that were highly rated by a user """
        
        # Restaurants that are more highly rated that the user's average star rating
        recommend_ids = []
        already_rated = list(orig_user.columns[(orig_user > 0).all()])
        
        for neighbour in neighbours:
            
            user_ratings = self.user_item_df.loc[neighbour[0]]
            user_mean_rating = user_ratings[user_ratings!=0].mean()
            restaurants_to_rec = list(user_ratings[user_ratings>user_mean_rating].index)
            restaurants_to_rec = np.setdiff1d(restaurants_to_rec,already_rated)
            zipped = zip(restaurants_to_rec, [neighbour[1]]*len(restaurants_to_rec))
            recommend_ids.extend(zipped)
        
        return recommend_ids    
    
    @staticmethod
    def scale_neighbour_distance(neighbour_tuple, to_add = 0):
        """ Scales the distance of a neighbour list of tuples such that the first element has a distance of 0"""
        
        scaled_neighbours = []
        neighbour_tuple.sort(key = operator.itemgetter(1))
        min_distance = neighbour_tuple[0][1]
        
        for business_id, distance in neighbour_tuple:
            new_distance = distance - min_distance + to_add
            scaled_neighbours.append((business_id,new_distance))
            
        scaled_neighbours.sort(key = operator.itemgetter(1))
        
        return scaled_neighbours
    
    def recommend(self, restaurant_names):
        """ Recommends restaurants"""
        
        # Hybrid recommender outputting list of restaurants
        given_restaurant_id1 = self.restaurant_name_df[self.restaurant_name_df['name']==restaurant_names[0]].index[0]
        given_restaurant_id2 = self.restaurant_name_df[self.restaurant_name_df['name']==restaurant_names[1]].index[0]
    
        content_neighbours_1 = self.get_restaurant_neighbours(given_restaurant_id1, K = 50)
        content_neighbours_2 = self.get_restaurant_neighbours(given_restaurant_id2, K = 50)
    
        # Based off of user's input restaurants, create mock user to compare to existing users
        mock_user = pd.DataFrame(0.0, columns=self.user_item_df.columns, index = ['mock_user'])
        mock_user[[given_restaurant_id1,given_restaurant_id2]] = 5.0
    
        # Neighbours to mock user from user-item filtering
        neighbours_to_mock_user = self.get_neighbour_user_ids(mock_user, K = 10)
        user_neighbours = self.get_restaurant_ids_from_neighbour_users(mock_user, neighbours_to_mock_user)
    
        #Combine all neighbours and remove duplicates and close name matches (for chain restaurants)
        combined_neighbours = self.scale_neighbour_distance(content_neighbours_1) + self.scale_neighbour_distance(content_neighbours_2) + self.scale_neighbour_distance(user_neighbours,to_add = 0.05)
        combined_neighbours = self.remove_dupe_neighbours(combined_neighbours)
    
        # Re-rank neighbours based off of user's input allergies and dietary accomodation requests
        allergy_adjusted_neighbours = self.re_rank_neighbours(combined_neighbours)
        
        return allergy_adjusted_neighbours
