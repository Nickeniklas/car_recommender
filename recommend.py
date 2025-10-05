import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

class ContentBasedRecommender:
    def __init__(self, data):
        self.data = data
        self.tfidf_matrix = None
        self.cosine_sim = None

    def preprocess(self):
        self.data['features'] = self.data['features'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.data['features'])

    def compute_similarity(self):
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def recommend(self, item_id, top_n=5):
        idx = self.data.index[self.data['id'] == item_id][0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        item_indices = [i[0] for i in sim_scores]
        return self.data.iloc[item_indices]

class HybridRecommender:
    def __init__(self, car_data_path):
        self.car_data = pd.read_csv(car_data_path)
        self.content_recommender = ContentBasedRecommender(self.car_data)