'''
import data here and have utility functions that could help
'''
from thefuzz import fuzz
from thefuzz import process
import pandas as pd
import numpy as np
import pickle

tracks = pd.read_csv('./data/track_features.csv')
feature_matrix = pd.read_csv('./data/feature_matrix.csv', index_col='id')
song_label = pd.read_csv('./data/song_label.csv', index_col='id')

#model
with open ('model/knn_recommender.pkl', 'rb') as file:
    model_knn = pickle.load(file)

#print(movies.head(5))

def movie_title_search(fuzzy_title, tracks):
    '''
    does a fuzzy search and returns best matched movie
    '''
    matches = process.extractBests(fuzzy_title, tracks, limit=1, scorer=fuzz.token_set_ratio)
    return matches

def name_to_id(title):
    '''
    converts movie title to id for use in algorithms
    '''
    song_label_2 = song_label.reset_index()
    song_label_2 = song_label_2.set_index('title')
    song_label_2.loc[title]['id']
    return id


# if __name__ == '__main__':
#         # fuzzy_matches = movie_title_search('star cars', song_label.set_index('id')['title'])
#     # print(fuzzy_matches)