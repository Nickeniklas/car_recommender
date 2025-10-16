import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

class ContentBasedRecommender:
    def __init__(self, cars_data):
        self.cars_data = cars_data
        self.tfidf_matrix = None
        self.car_indices = None

    def fit(self):
        """ Data preprocess and feature extraction. """
        self.cars_data['Features'] = self.cars_data['Features'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.cars_data['Features'])
        self.car_indices = pd.Series(self.cars_data.index, index=self.cars_data['Make Model Year']).drop_duplicates()

    def recommend(self, car, n=10):
        """ 
        Return top-n similiar cars

        Parameters:
            car = make, model and year (small letter and space ex. toyota sienna 2020).
            n = amount of recommends.
        """
        if car not in self.car_indices: return [] 
        value = self.car_indices[car]
        idx = value.iloc[0] if isinstance(value, pd.Series) else int(value) # single match and multiple matches handled
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        scores = pd.Series(sim_scores, index=self.cars_data['carID'])
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9) # normalize CB scores
        return scores.sort_values(ascending=False).head(n)
    
class CollaborativeRecommender:
    def __init__(self, ratings_data):
        self.ratings_data = ratings_data
        self.user_item_matrix = None
        self.user_mapper = None
        self.car_mapper = None
        self.nmf_model = None

    def fit(self):
        """ Train NMF on user-item matrix. """
        # user item matrix
        self.user_item_matrix = self.ratings_data.pivot_table(index='userID', columns='carID', values='Rating').fillna(0)
        # map user ID with row index and car ID with column index.
        self.user_mapper = {uid: i for i, uid in enumerate(self.user_item_matrix.index)}
        self.car_mapper = {mid: i for i, mid in enumerate(self.user_item_matrix.columns)}
        # Initialize the NMF model
        self.nmf_model = NMF(n_components=20, init='random', random_state=42, max_iter=500)
        self.nmf_model.fit(self.user_item_matrix)

    def recommend(self, userID, n=10):
        """Predict scores for all cars."""
        if userID not in self.user_mapper:
            return pd.Series()  # No data for this user
        user_idx = self.user_mapper[userID]
        user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        user_P = self.nmf_model.transform(user_vector)
        item_Q = self.nmf_model.components_
        scores = np.dot(user_P, item_Q).flatten()
        scores = pd.Series(scores, index=self.user_item_matrix.columns)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9) # Normalize CF scores
        return scores.sort_values(ascending=False).head(n)

class HybridRecommender:
    def __init__(self, cars_data_path, ratings_data_path):
        # data
        self.cars_data = pd.read_csv(cars_data_path)
        self.ratings_data = pd.read_csv(ratings_data_path)
        # recommenders
        self.cb_model = ContentBasedRecommender(self.cars_data)
        self.cf_model = CollaborativeRecommender(self.ratings_data)

    def fit(self):
        self.cb_model.fit()
        self.cf_model.fit()

    def id_to_title(self, score_series, top_n=None):
        """ Map carID → 'Make Model Year" for readability. """
        # dataframe with car titles 
        df = self.cars_data[['carID', 'Make Model Year']].set_index('carID')
        merged = df.join(score_series.rename('Score'), how='inner')
        if top_n:
            merged = merged.sort_values('Score', ascending=False).head(top_n)
        return merged

    def recommend(self, userID, car, n=10, alpha=0.5):
        """ Combine collaborative and content-based predictions. """
        cf_scores = self.cf_model.recommend(userID)
        cb_scores = self.cb_model.recommend(car)
        hybrid_score  = alpha * cf_scores + (1 - alpha) * cb_scores 
        top_ids = hybrid_score.nlargest(n).index
        recs = self.cars_data[self.cars_data['carID'].isin(top_ids)]['Make Model Year'].tolist()
        return recs

if __name__ == "__main__":
    # dataset paths
    cars_data_path = "./data/df_cars_clean.csv"
    ratings_data_path = "./data/df_ratings_clean.csv"

    # initialize hybrid recommender
    hybrid = HybridRecommender(cars_data_path, ratings_data_path)
    hybrid.fit()

    # test seeds
    car = 'volkswagen passat 2.0 tdi sel 2012'
    userID = 27583
    
    # content based
    cb_scores = hybrid.cb_model.recommend(car)
    print("Content-based recommendations:\n", hybrid.id_to_title(cb_scores, 5))

    # collaborative
    cf_scores = hybrid.cf_model.recommend(userID)
    print("Collaborative recommendations:\n", hybrid.id_to_title(cf_scores, 5))

    # hybrid 
    hybrid_scores = hybrid.recommend(userID, car, n=10, alpha=0.5)
    print("Hybrid recommendations:\n", hybrid_scores)

    
