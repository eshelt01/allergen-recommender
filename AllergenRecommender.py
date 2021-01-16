# -*- coding: utf-8 -*-
"""

@author: Erin S
"""

from hybridRecommender.hybridRecommender import hybridRecommend

# Reference allergen and dietary key list options
allergen_key_master_list = ['milk','egg','fish','crustacean','tree_nut','peanut','wheat','soy']
dietary_key_master_list = ['vegan', 'vegetarian', 'celiac']


#User input -> allergies and dietary accomodations
allergen_key_list = ['peanut']
dietary_key_list = ['vegetarian']

#User input -> two restaurants that they already enjoy
restaurant_names = ['STACK','Pizzeria Libretto']

# Content-based neighbours from given restaurants

recommender = hybridRecommend(allergen_key_list, dietary_key_list)
neighbours = recommender.recommend(restaurant_names)

# Print top 20 recommendations
recommender.print_recommendations(neighbours)
