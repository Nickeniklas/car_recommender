import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

class ContentBasedRecommender:
    def __init__(self, cars_data):
        self.cars_data = cars_data
        self.tfidf_matrix = None
        self.cosine_sim = None

    def fit(self):
        self.cars_data['Features'] = self.cars_data['Features'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.cars_data['Features'])
        self.car_indices = pd.Series(self.cars_data.index, index=self.cars_data['Make Model']).drop_duplicates()
        #print("Fit:", self.car_indices)
        #print(self.tfidf_matrix.shape) 
    
    def recommend(self, car, n=10):
        if car not in self.car_indices: return []
        print("Content based recommends for", car)
        idx = int(self.car_indices[car].iloc[0])
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_indices = sim_scores.argsort()[-n-1:-1][::-1]
        return self.cars_data['Make Model'].iloc[sim_indices].tolist()

class HybridRecommender:
    def __init__(self, cars_data_path, ratings_data_path):
        self.car_data = pd.read_csv(cars_data_path)
        self.rating_data = pd.read_csv(ratings_data_path)
        self.content_recommender = ContentBasedRecommender(self.car_data)
        
        
if __name__ == "__main__":
    # get datasets
    cars_data_path = "./data/df_cars_clean.csv"
    ratings_data_path = "./data/df_ratings_clean.csv"
    cars_data = pd.read_csv(cars_data_path)
    ratings_data = pd.read_csv(ratings_data_path)
    #print("cars data: \n", cars_data)
    #print("ratings data: \n", ratings_data)

    # test content based recommender
    recommender = ContentBasedRecommender(cars_data)
    recommender.fit()

    print(recommender.recommend('toyota sienna', n=5))
    
