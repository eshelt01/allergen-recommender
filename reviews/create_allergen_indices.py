# -*- coding: utf-8 -*-
"""
@author: Erin S
"""

import pandas as pd
import Review as rev
from statistics import mean

# Script to create and save the allergen indices for each restaurant

review_df = pd.read_pickle('C:/Users/Ed/Documents/allergen-recommender/data/restaurant_reviews.pkl')
review_df['date'] = pd.to_datetime(review_df['date'])

# Major food allergens:
# Foods/phrases of interest for each of the top 8 allergens sourced from "https://foodallergycanada.ca"

Allergens_Contains = {'Milk': ['milk','cream','cheese','yogurt','gelato','butter'],
                      'Eggs': ['egg','marzipan','meringue','eggnog','mayonnaise','lysozyme','albumin'],
                      'Fish': ['fish','bass','cod','salmon','tilapia','halibut','haddock','trout','anchovy','carp','flounder','grouper','eel','sushi','sashimi','mackerel','mahi','pollock','sardine','shark','swordfish','turbot','sturgeon','caviar','ceviche','surimi','tarama','lox'],
                      'Crustaceans': ['lobster','crab','shrimp'],
                      'Tree_nuts': ['almond','brazil nut','cashew','chestnut','hazelnut','hickory','macadamia','pecan','pine nut','pistachio','walnut','pesto','nutella','praline'],
                      'Peanuts' : ['peanut','beer nut','arachis oil','valencias','szechuan'],
                      'Wheat' : ['wheat','flour','couscous','bread','spelt'],
                      'Soy' : ['soy','soybean','soya','tempeh','tofu','miso','natto','edamame','teriyaki']}

Allergens_Free_From = {'Milk': ['dairy free','milk free','lactose free'],
                       'Eggs': ['egg free'],
                       'Fish' : ['fish free'],
                       'Crustaceans' : ['shellfish free','crustacean free'],
                       'Tree_nuts': ['nut free'],
                       'Peanuts' : ['nut free', 'peanut free'],
                       'Wheat' : ['wheat free', 'gluten free'],
                       'Soy': ['soy free', 'soya free']}


Dietary_Accomodations = {'Allergy' : ['dietary','allergic','allergy','allergies'],
                        'Vegan' : ['vegan'],
                        'Vegetarian' : ['vegetarian'],
                        'Celiac': ['celiac', 'gluten']}

# Optional time cutoff for which reviews to include
#time_cutoff = '2016-01-01'

def unique_business_ids(review_df, time_cutoff = '2000-01-01'):    
    
    unique_business_id = review_df['business_id'][review_df['date']>time_cutoff].unique()
    return unique_business_id           


def find_sentences_sentiments(allergen_key, review_text, phrase_dict):
    
    found_sentiments = []
    found_sentences = []
    
    for review in review_text:

            current_Review  = rev.Review(text = review, phrase_dict = phrase_dict)
            sentence = current_Review.find_sentences(allergen_key)
            sentiment = current_Review.calc_sentence_sentiments()

            if sentence:
                found_sentiments.append(sentiment)
                found_sentences.append(sentence)
                
                
    return found_sentences, found_sentiments



def create_index_df(unique_restaurants, df, allergen_key_list, phrase_dict, review_df, use_sentiments = False):
       
    for count,business_id in enumerate(unique_restaurant_ids):
        
        fraction_list = []
        review_text = review_df['text'][review_df['business_id']==business_id]

        for allergen_key in allergen_key_list:
            # Sentences and sentiments corresponding to specific allergen keywords
            found_sentences, found_sentiments = find_sentences_sentiments(allergen_key, review_text, phrase_dict) 
            fraction = len(found_sentiments)/len(review_text)        
            fraction_list.append(fraction)

            if use_sentiments:
                
                 if found_sentiments:
                    fraction_list.append(mean(found_sentiments))
                 else:
                    fraction_list.append(None)
          
        df.loc[business_id] = fraction_list  

    return df

# Create indices for 'contains', 'free from' and 'dietary accomodations'
unique_restaurant_ids = unique_business_ids(review_df)

# Create allergen contains index
df_contains = pd.DataFrame(columns = ['milk_fraction','egg_fraction','fish_fraction','crustacean_fraction','tree_nut_fraction','peanut_fraction','wheat_fraction','soy_fraction'])
args = {'unique_restaurants': unique_restaurant_ids , 'df': df_contains, 'allergen_key_list': Allergens_Contains.keys(), 'phrase_dict': Allergens_Contains, 'review_df' :review_df}

df_contains = create_index_df(**args)
#df_contains.to_pickle('./data/df_allergen_contains.pkl') 

# Create allergen free from index
df_free_from = pd.DataFrame(columns = ['milk_fraction','milk_sentiment','egg_fraction','egg_sentiment','fish_fraction','fish_sentiment','crustacean_fraction','crustacean_sentiment','tree_nut_fraction','tree_nut_sentiment','peanut_fraction','peanut_sentiment','wheat_fraction','wheat_sentiment','soy_fraction','soy_sentiment'])
args = {'unique_restaurants': unique_restaurant_ids , 'df': df_free_from, 'allergen_key_list': Allergens_Free_From.keys(), 'phrase_dict': Allergens_Free_From, 'review_df' :review_df, 'use_sentiments':True}

df_free_from = create_index_df(**args)
#df_free_from.to_pickle('./data/df_allergen_free_from.pkl')  

# Create dietary accomodations index
df_dietary = pd.DataFrame(columns = ['allergy_fraction','allergy_sentiment','vegan_fraction','vegan_sentiment','vegetarian_fraction','vegetarian_sentiment','celiac_fraction','celiac_sentiment'])
args = {'unique_restaurants': unique_restaurant_ids , 'df': df_dietary, 'allergen_key_list': Dietary_Accomodations.keys(), 'phrase_dict': Dietary_Accomodations, 'review_df' :review_df, 'use_sentiments':True}  
 
df_dietary = create_index_df(**args) 
#df_dietary.to_pickle('./data/df_dietary_accomodations.pkl')   
