# -*- coding: utf-8 -*-
"""
@author: Erin S
"""
import nltk
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from statistics import mean 


class Review:
    """Review class for text processing.
    """
    
    def __init__(self, text, phrase_dict, allergen_key = None):
        self.stop_words = set(stopwords.words("english"))
        self.text = text
        self.allergen_key = allergen_key 
        self.phrase_dict = phrase_dict
        self.tokens = []        
        self.lemmatized = []      
        self.sentences = ""
        self.sentiment = [] 

    def clean_text(self):
    
        self.text  = re.sub('\n',' ',self.text)
        self.text  = re.sub('////',' ',self.text)
        self.text  = re.sub('\\\\',' ',self.text)
        return


    def tokenize_text(self):
         
        self.tokens = word_tokenize(self.text)
        return        
 
    
    def clean_tokens(self):
        # To lowercase and remove stopwords
        lower_tokens = [w.lower() for w in self.tokens]
        clean_tokens = [w.strip() for w in lower_tokens if len(w) >0 and w not in string.punctuation and w not in self.stop_words]   
        return clean_tokens


    def get_wordnet_tags(self, token_tuple):
        # Get associated part-of-speech tags for words
        tag_first_char = token_tuple[1][0]
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB,"R": wordnet.ADV}

        try:
            wordnet_tag = tag_dict.get(tag_first_char.upper())

        except:

            wordnet_tag = None

        return (token_tuple[0],wordnet_tag)


    def lemmatize_text(self,wordnet_pos_tags):
        
        lemmatized_text = []
        lemmatizer = WordNetLemmatizer()
        
        for word, tag in wordnet_pos_tags:
                if tag:
                    lemmatized_text.append(lemmatizer.lemmatize(word, tag))

                else:        
                    lemmatized_text.append(word)

        self.lemmatized = " ".join(lemmatized_text)
        return  
   


    def process_text(self):    
        # Clean, tokenize, and lemmatize text
        self.clean_text()
        self.tokenize_text()
        self.tokens = self.clean_tokens()
        nltk_pos_tags_list  = nltk.pos_tag(self.tokens) 

        wordnet_pos_tags = [self.get_wordnet_tags(token_tuple) for token_tuple in nltk_pos_tags_list]
        self.lemmatize_text(wordnet_pos_tags) 
           
        return 


    def find_sentences(self, key = None):
        # Find sentences containing keyword
        if key:
            self.allergen_key = key       
        
        if self.allergen_key:
            
            key = self.allergen_key
            self.clean_text()
            keywords = self.phrase_dict[key]
            split_sentence = nltk.sent_tokenize(self.text)
            allergy_sentences = [sentence for sentence in split_sentence if any(word in sentence for word in keywords)]      
            self.sentences = allergy_sentences   
              
        return allergy_sentences
    
    
    def calc_sentence_sentiments(self):
        # Use VADER to find sentiment (pos,neu,neg) of keyword sentences
        self.sentiment = []
        
        if len(self.sentences)>0:
            
            sid_obj = SentimentIntensityAnalyzer() 
            sents = []

            for sentence in self.sentences: 
                sentiment_dict = sid_obj.polarity_scores(sentence) 
               # max_key = max(sentiment_dict, key=sentiment_dict.get)

                if sentiment_dict['compound'] >= 0.05:     # Positive polarity      
                    val = 1
                    sents.append(val)

                elif sentiment_dict['compound'] <= - 0.05: # Negative polarity
                    val = -1
                    sents.append(val)
                   
                else:                                      # Neutral polarity
                    val = 0
                    sents.append(val)
            
            if len(sents)>=1:
                self.sentiment =  mean(sents)              # Mean sentiment -> overall review sentiment                
        
            else:
                self.sentiment = []
        
        return self.sentiment
    
