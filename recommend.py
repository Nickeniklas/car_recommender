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
        """ Data preprocess and feature extraction. """
        self.cars_data['Features'] = self.cars_data['Features'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.cars_data['Features'])
        self.car_indices = pd.Series(self.cars_data.index, index=self.cars_data['Make Model Year']).drop_duplicates()
    
    def recommend(self, car, n=10):
        """ 
        Content based recommendation based on feature extraction. 

        Parameters:
            car = make and model (small letter and space),
            n = number of recommendations to return.
        """
        if car not in self.car_indices: return []
        value = self.car_indices[car]
        idx = value.iloc[0] if isinstance(value, pd.Series) else int(value) # single match and multiple matches handled
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_indices = sim_scores.argsort()[-n-1:-1][::-1]
        return self.cars_data['Make Model Year'].iloc[sim_indices].tolist()

class HybridRecommender:
    def __init__(self, cars_data_path, ratings_data_path):
        self.car_data = pd.read_csv(cars_data_path)
        self.rating_data = pd.read_csv(ratings_data_path)
        self.content_recommender = ContentBasedRecommender(self.car_data)
        self.nmf_model = None; self.user_item_matrix = None; self.user_mapper = None
        self.movie_mapper = None; self.movie_inv_mapper = None
    def fit(self):
        """ Non-negative Matrix Factorization and fit content based recommender."""
        self.content_recommender.fit()
        # user item matrix
        self.user_item_matrix = self.rating_data.pivot_table(index='userID', columns='carID', values='Rating').fillna(0)
        # map user ID with row index and car ID with column index.
        self.user_mapper = {uid: i for i, uid in enumerate(self.user_item_matrix.index)}
        self.movie_mapper = {mid: i for i, mid in enumerate(self.user_item_matrix.columns)}
        # Initialize the NMF model
        self.nmf_model = NMF(n_components=20, init='random', random_state=42, max_iter=500)
        self.nmf_model.fit(self.user_item_matrix)
    
    def get_hybrid_scores(self, userID, car, alpha=0.5):
        """Compute weighted hybrid scores combining collaborative and content-based results."""
        # --- Collaborative predictions ---
        if userID not in self.user_mapper:
            return pd.Series()  # No data for this user

        user_idx = self.user_mapper[userID]
        user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        user_P = self.nmf_model.transform(user_vector)
        item_Q = self.nmf_model.components_
        predicted_scores = np.dot(user_P, item_Q).flatten()
        cf_scores = pd.Series(predicted_scores, index=self.user_item_matrix.columns)

        # --- Normalize CF scores ---
        cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min())

        # --- Content-based similarity scores ---
        if car not in self.content_recommender.car_indices:
            cb_scores = pd.Series(0, index=cf_scores.index)
        else:
            car_idx = self.content_recommender.car_indices[car]
            if isinstance(car_idx, pd.Series): car_idx = car_idx.iloc[0]
            sim_scores = cosine_similarity(self.content_recommender.tfidf_matrix[car_idx], 
                                        self.content_recommender.tfidf_matrix).flatten()
            cb_scores = pd.Series(sim_scores, index=self.car_data.index)
            cb_scores.index = self.car_data['carID']  # align by carID
            cb_scores = cb_scores.reindex(cf_scores.index).fillna(0)
            cb_scores = (cb_scores - cb_scores.min()) / (cb_scores.max() - cb_scores.min())

        # --- Combine ---
        final_scores = alpha * cf_scores + (1 - alpha) * cb_scores
        return final_scores
    
        
    def recommend(self, userID, car, n_recs=10, alpha=0.5):
        """Return top hybrid recommendations for user and car."""
        final_scores = self.get_hybrid_scores(userID, car, alpha)
        if final_scores.empty:
            return self.content_recommender.recommend(car, n_recs)

        rated = self.rating_data[self.rating_data['userID'] == userID]['carID']
        final_scores = final_scores.drop(index=rated, errors='ignore')

        top_ids = final_scores.nlargest(n_recs).index.tolist()
        hybrid_recs = self.car_data[self.car_data['carID'].isin(top_ids)]['Make Model Year'].tolist()
        return hybrid_recs

if __name__ == "__main__":
    # get datasets
    cars_data_path = "./data/df_cars_clean.csv"
    ratings_data_path = "./data/df_ratings_clean.csv"
    cars_data = pd.read_csv(cars_data_path)
    ratings_data = pd.read_csv(ratings_data_path)
    #print("cars data: \n", cars_data)
    #print("ratings data: \n", ratings_data)

    # test
    car_make_model = 'mercedes-benz amg c 43 2018'
    userID = 27583
    test_to_run = 'hybrid' # change between: 'hybrid' and 'content'

    if test_to_run == 'content':
        contentBasedRecommender = ContentBasedRecommender(cars_data)
        contentBasedRecommender.fit()
        recs = contentBasedRecommender.recommend(car_make_model, 5)
        print(f"Content based Recommendations for {car_make_model}:\n{recs}")

    elif test_to_run == 'hybrid':
        hybridRecommender = HybridRecommender(cars_data_path, ratings_data_path)
        hybridRecommender.fit()
        recs = hybridRecommender.recommend(userID, car_make_model, 5)

        print(f"Hybrid Recommendations for user {userID} and car {car_make_model}:\n {recs}")
    
