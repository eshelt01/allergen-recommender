# -*- coding: utf-8 -*-
"""

@author: Erin S
"""
# Script for processing YELP dataset

import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer

def normalize_str_dict(df,column_names):
    # Normalize columns where row elements are dictionaries
    for column in column_names:
        
        if column in df.columns:
        
            col_holder = df[column]          
            col_holder= col_holder.apply(lambda x: "None" if pd.isna(x) else x) # Formatting for literal_eval
            col_holder = col_holder.apply(ast.literal_eval)
            col_holder= col_holder.apply(lambda x: {} if pd.isna(x) else x)     # Empty braces for json_normalize
            normalized_col = pd.json_normalize(col_holder)
            df = df.join(normalized_col, how = 'outer' )           
            df = df.drop(columns = column)
        else:
            print ("Warning: {} does not exist in {} ".format(column,df))
    
    return df 

#----------------------------------------------------------------------------#
# Import business information from restaurants: keep only restaurants in the GTA cities, with > 10 reviews

city_list = ['Toronto','Mississauga','Etobicoke','Vaughan','Richmond Hill','Thornhill','Brampton','Etobicoke', 'Scarborough','Whitby','North York']
min_review_count = 10

business_json_path = './data/yelp_academic_dataset_business.json'

df_business= pd.read_json(business_json_path, lines = True)
df_business = df_business[df_business['is_open'] == 1]
df_business_GTA = df_business[df_business['city'].isin(city_list)]

# Only keep restaurants with reviews> min review count
df_GTA_restaurant = df_business_GTA[(df_business_GTA['categories'].str.contains('Restaurants|Food',na = False))]
df_GTA_restaurant = df_GTA_restaurant[~df_GTA_restaurant['categories'].str.contains('Grocery',na = False)]
df_GTA_restaurant = df_GTA_restaurant[df_GTA_restaurant['review_count']>min_review_count]  

# Drop duplicates and reset index
df_GTA_restaurant = df_GTA_restaurant.loc[df_GTA_restaurant.astype(str).drop_duplicates().index]
df_GTA_restaurant = df_GTA_restaurant.reset_index()
df_GTA_restaurant.to_pickle("./data/restaurant_business_cleaned.pkl")

# Save name/location separately
restaurant_name_info = df_GTA_restaurant[['business_id','name','address','stars','review_count']]
restaurant_name_info = restaurant_name_info.set_index('business_id')
restaurant_name_info.to_pickle('./data/restaurant_name_info.pkl')

# Wish to only keep relevant columns (categories,attributes,etc.) for use in recommender algorithm
# Convert attributes to columns
df_GTA_restaurant['attributes']= df_GTA_restaurant['attributes'].apply(lambda x: {} if pd.isna(x) else x)
normalized_attributes_df = pd.json_normalize(df_GTA_restaurant['attributes'])
df_GTA_restaurant = df_GTA_restaurant.join(normalized_attributes_df, how = 'outer' )

# Drop columns with large amounts of missing data
df_GTA_restaurant.drop(columns = ['level_0','index','city','longitude','latitude','name','address','DietaryRestrictions','RestaurantsCounterService','WiFi','Alcohol','NoiseLevel','BYOB','GoodForDancing','DogsAllowed','WheelchairAccessible', 'RestaurantsTableService','attributes','postal_code','BestNights','state','HasTV','is_open','hours','BikeParking','BusinessParking','Caters','AgesAllowed','DriveThru','Corkage','Music','AcceptsInsurance','BusinessAcceptsBitcoin','BusinessAcceptsCreditCards','ByAppointmentOnly','HappyHour','CoatCheck','Smoking'], inplace=True)

# Certain columns require normalization
columns_to_normalize = ['Ambience','GoodForMeal']
df_GTA_restaurant = normalize_str_dict(df_GTA_restaurant,columns_to_normalize)

# Categories column is in alternate format requiring a different form of normalization
mlb = MultiLabelBinarizer()

df_GTA_restaurant['categories'] = df_GTA_restaurant['categories'].replace(np.nan, 'Unknown', regex=True) # If missing values set to 'Unknown' for MLB compatibilty
df_GTA_restaurant['categories'] = df_GTA_restaurant['categories'].apply(lambda x: x.split(','))

expanded_categories = pd.DataFrame(mlb.fit_transform(df_GTA_restaurant['categories']), columns = mlb.classes_, index = df_GTA_restaurant.index)

# Map string information in specific columns to numerical info
columns = ['RestaurantsTakeOut','GoodForKids' ,'RestaurantsPriceRange2','RestaurantsAttire','RestaurantsDelivery','RestaurantsGoodForGroups' ,'RestaurantsReservations','OutdoorSeating']                                 
df_GTA_restaurant = pd.get_dummies(data = df_GTA_restaurant,columns = columns)

# Remove "none" dummy columns
df_GTA_restaurant.drop(columns = ['GoodForKids_None','RestaurantsPriceRange2_None','RestaurantsPriceRange2_None','RestaurantsAttire_None','RestaurantsGoodForGroups_None','RestaurantsReservations_None','RestaurantsReservations_None','OutdoorSeating_None'],inplace = True)

# Keep only most frequent/relevant categories
category_freq = pd.DataFrame(columns = ['frequency'])

for column in expanded_categories:
    category_freq.loc[column] = np.mean(expanded_categories[column])

categories_to_keep = category_freq[category_freq['frequency']>0.03].index 
expanded_categories = expanded_categories[categories_to_keep]

expanded_categories.drop(columns = ['Restaurants',' Restaurants','Food',' Food'], inplace = True) # Not useful identifiers
df_GTA_restaurant_processed = df = pd.concat([expanded_categories,df_GTA_restaurant], axis=1)
df_GTA_restaurant_processed = df_GTA_restaurant_processed.set_index('business_id')
df_GTA_restaurant_processed.drop(columns = ['categories'],inplace = True)

# Final processing and saving
df_GTA_restaurant_processed = df_GTA_restaurant_processed*1                                          # Convert True/False to 1/0
df_GTA_restaurant_processed = df_GTA_restaurant_processed.fillna(df_GTA_restaurant_processed.mean()) # Fill missing values with mean of column
df_GTA_restaurant_processed.to_pickle('./data/restaurant_categories_cleaned.pkl')

#----------------------------------------------------------------------------#
# Import reviews from relevant restaurants
review_json_path = './data/yelp_academic_dataset_review.json'
size = 10000

chunk_list =[]

for chunk_review in pd.read_json(review_json_path, lines = True, dtype={'review_id':str,'user_id':str,'business_id':str,'stars':int,'date':str,'text':str,'useful':int,'funny':int,'cool':int},chunksize = size,nrows = 10000000):
  
    chunk_review = chunk_review.drop(['useful','funny','cool'], axis=1)
    chunk_review = chunk_review.rename(columns={'stars': 'review_stars'})    # Renaming column name to avoid conflict
    
    # Only keep reviews related to relevant reviews 
    chunk_merged = pd.merge(df_GTA_restaurant['business_id'], chunk_review, on='business_id', how='inner')
    chunk_list.append(chunk_merged)
    
review_df = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)   

# Some reviewers have reviewed a restaurant multiple times -> In this case only use the most recent review.
multiple = review_df[review_df.duplicated(subset=['business_id','user_id'], keep = False)]
multiple = multiple.groupby('user_id').count().reset_index()
multiple_reviewers = multiple['user_id']

for index, user_id in enumerate(multiple_reviewers):
    reviewer = review_df[review_df['user_id'] == user_id]
    reviews_sorted = reviewer[reviewer.duplicated(['business_id'],keep = False)].sort_values(by = 'date')
    review_df.drop(reviews_sorted.index[0:len(reviews_sorted.index)-1],inplace = True) 

review_df.to_pickle("./data/restaurant_reviews.pkl")
    
#----------------------------------------------------------------------------#
# Import users

user_json_path = './data/yelp_academic_dataset_user.json'
size = 10000

chunk_list =[]

for chunk_user in pd.read_json(user_json_path, lines = True,chunksize = size ,nrows = 10000000):
    chunk_list.append(chunk_user)
    
df_user = pd.concat(chunk_list, ignore_index = True, join ='outer', axis  =0) 
df_user.rename(columns={'review_count':'user_review_count','average_stars':'user_average_stars'}, inplace = True)
df_user = df_user.drop(['name','yelping_since','elite','funny','cool','friends','fans','compliment_hot','compliment_more','compliment_profile','compliment_cute','compliment_list','compliment_note','compliment_plain','compliment_cool','compliment_funny','compliment_writer','compliment_photos'], axis=1)

# Only keep users who have reviews our relelvant restaurants
relevant_users = df_user[df_user['user_id'].isin(review_df['user_id'])]
relevant_users.to_pickle("./data/user.pkl")

# May need to run separately due to memory requirements
# Create and save user-item matrix for subset of users for collaborative filtering
user_df = pd.read_pickle('./data/user.pkl')
review_df = pd.read_pickle('./data/restaurant_reviews.pkl')

review_df = review_df.pivot(index ='user_id', columns ='business_id', values =['review_stars']) 
review_df.columns = review_df.columns.droplevel(0)
review_df = review_df.fillna(0)
ids_to_keep = []

for user_id, current_user in review_df.iterrows():
  
    if (len(current_user.to_numpy().nonzero()[0])) >5:   # Only keep if users have reviewed more than 5 of our relevant GTA restaurants
        ids_to_keep.append(user_id)

review_df = review_df.loc[ids_to_keep]    
review_df.to_pickle('./data/user_item_review_stars.pkl')



