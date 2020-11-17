# allergen-recommender
Hybrid-recommender providing individualized restaurant recommendations for those with food allergies and strong dietary preferences for restaurants in the Greater Toronto Area (GTA). 

## Description
  This project intends to provide a recommendation system to reduce the time/effort associated with finding relevant and appropriate restaurants for those with food allergies. Finding restaurants that have positive reviews in terms of allergen-specific care and low frequency of the allergen in the menu dishes can be extremely time-consuming for an individual consumer. 

  This recommender a content-based and user-item collaborative filtering-based hybrid recommender system using restaurant data from the most recent YELP challenge dataset. The recommender uses NLP to adjust the ranking of restaurants based on user-provided food allergies and dietary preferences. The top 8 food allergens (milk, crustaceans, wheat, tree nuts, peanuts, soy, fish, eggs) and specific dietary preferences (vegetarian, vegan, gluten-free) are supported. 

## Files

### Data pre-processing

- dataset_cleaning: script for loading YELP dataset, cleaning, and saving to .pkl files (offline)

### NLP
- Review.py: class for basic text cleaning, lemmatization, and sentiment analysis for reviews
- TFIDF_process.py: script for term frequency–inverse document frequency processing on reviews
- create_allergen_indices.py: script to create dataframes/.pkl for custom 'allergen index' for each restaurant

### Recommender
- hybrid_recommend.py: module for recommender functions
- AllergenRecommender.py & AllergenRecommender.ipynb: script/notebook for final recommender

