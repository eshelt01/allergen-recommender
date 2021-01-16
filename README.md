# allergen-recommender
A hybrid recommender providing individualized restaurant recommendations for those with food allergies and strong dietary preferences for restaurants in the Greater Toronto Area (GTA). 

 ## Description
   This project intends to provide a recommendation system to reduce the time and effort associated with finding relevant and appropriate restaurants for those with food allergies. Finding restaurants that have positive reviews in terms of allergen-specific care and low frequency of the allergen in the menu dishes can be extremely time-consuming for an individual consumer. 

   This recommender a content-based and user-item collaborative filtering-based hybrid recommender system using restaurant data from the most recent YELP challenge dataset. The recommender uses NLP to adjust the ranking of restaurants based on user-provided food allergies and dietary preferences. The top 8 food allergens (milk, crustaceans, wheat, tree nuts, peanuts, soy, fish, eggs) and specific dietary preferences (vegetarian, vegan, gluten-free) are supported. 


## Files

### Data pre-processing

- dataset_cleaning: script for loading YELP dataset, cleaning, and saving to .pkl files 

### NLP
- Review.py: class for basic text cleaning, lemmatization, and sentiment analysis for reviews
- TFIDF_process.py: script for term frequency–inverse document frequency processing on reviews
- create_allergen_indices.py: script to create dataframes for custom 'allergen index' for each restaurant

### Recommender
- hybridRecommender.py: class for recommender functions
- AllergenRecommender.py : main script for final recommender

 ## Example Case:

 A new user supplies two restaurants that they already enjoy, here we will use 'Boxcar Social' and 'Harvest Green'. We construct a ranked list of recommended restaurants using AllergenRecommender.
 
 In this example the top 5 are:
 
 Rank | Restaurant
------------ | -------------
1 | Melodie's
2 | Page One
3 | Gallo Churrasqueira
4 | Mr. Greek
5 | Epic Pita


Now, if we run the recommendation with a specified tree nut allergy, the top 5 restaurants are now:

 Rank | Restaurant
------------ | ------------- 
1 | Melodie's
2	| Gallo Churrasqueira
3	| Epic Pita
4	| Galito's Flame Grilled Chicken
5	| Wingporium

We can visualize the effect of the tree nut allergen filtering on the recommendations by looking at the change in the metric used to compute ranking (similarity distance) between the two cases (no allergy/ tree nut allergy) for the top restaurants:

![Tree_nut_allergy_ranking](https://user-images.githubusercontent.com/66339416/103049553-12d41d00-4560-11eb-9b51-62b923bbe487.png)


Typically several restaurants become more suitable (negative change), while other become less suitable (positive change). Additionally many restaurants show no change to their ranking metric (ie. no mention of specific allergen or of any allergy related issues in reviews). 

In this case we see that a top 5 restaurant (Page One) drops dramatically in the rankings and would no longer be recommended to a user with a tree nut allergy.

By looking at the created allergen indices, we can see that 9.6 % of the reviews for this restaurant contained keywords that are associated with a tree nut allergy.
Inspecting the text of these reviews shows that the restauraunt features several products that contain almonds, pecans, and Nutella:

-----------------------------------------------------------------------------------------------------------------------------------------------------------
“We finished the meal with a latte and a **pecan** square. The **pecan** square was really good. Sweet, melt-in-your-mouth, delicate buttery crust. “

" My first trip I had a matcha green tea latte with **almond** milk and some vegan energy bites and my second stop was for breakfast where I had a dirty chai latte with **almond** milk with a breakfast sandwich.” 

“Both of us got **Nutella** lattes since we've never tried anything like that before.  Thinking that it will be a flavored latte, I didn't expect the drink to be so bad ... As for the **Nutella**, I could not taste any **Nutella** and it wasn't until I stirred that I realized that all the **Nutella** was at the bottom.

“We had **Nutella** Latte and Mocha and both were very milky.”

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
## Allergen Filtering
   * The allergen filtering punishes restaurants that contain reviews with a high frequency of food items that contain the allergen, with the assumption that a user may want to avoid a restaurant whose popular dishes contain said allergen and the risk of cross-contamination is higher.
   * The filtering also looks at associated phrases in reviews that indicate 'free-from' dishes (ie. 'lactose-free', 'nut-free'), and the sentiments of those phrases. In this way a restaurant whose reviews contain a positive sentiment phrase like 'the nut-free options here are fantastic!', will be adjusted towards a better ranking. 
   * Finally, the filtering looks at generic allergen/dietary phrasing and the sentiments of those phrases. Here the phrases are associated with a reviewers generic experience with allergens and dietary accomodations, and not a specific allergen. This includes phrases like 'allergy', 'dietary needs' , etc. Again, ranking is adjusted based on phrase sentiment.
   
  * The relative contribution of these factors in the adjustment of the restaurant rankings can be manually tuned based on user-preferences.
 
