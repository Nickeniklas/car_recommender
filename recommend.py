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
        self.car_indices = pd.Series(self.cars_data.index, index=self.cars_data['Make Model']).drop_duplicates()
    
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
        return self.cars_data['Make Model'].iloc[sim_indices].tolist()

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
    
    def recommend(self, userID, car, n_recs=10):
        """
        Hybrid recommendation combining content-based and collaborative filtering.

        Parameters:
            userID: ID of the user for collaborative filtering
            car: car name (used for content-based recommendations)
            n_recs: number of recommendations to return

        Returns:
            List of recommended car titles
        """
        # content-based recommendations
        content_recs = self.content_recommender.recommend(car, n_recs)
        collaborative_recs = []
        if userID in self.user_mapper:
            # map userID to the row index in the user-item matrix
            userIDx = self.user_mapper[userID]
            # get the users ratings
            user_vector = self.user_item_matrix.iloc[userIDx].values.reshape(1, -1)
            # user vector latent space using trained NMF model
            user_P = self.nmf_model.transform(user_vector)
            # item latent feature matrix
            item_Q = self.nmf_model.components_
            # predicted scores for all items (dot product of user and item latent features)
            predicted_scores = np.dot(user_P, item_Q).flatten()
            # convert predicted scores to a pandas Series with car IDs as the index
            scores_series = pd.Series(predicted_scores, index=self.user_item_matrix.columns)
            # remove items the user has already rated
            rated_cars = self.rating_data[self.rating_data['userID'] == userID]['carID']
            scores_series = scores_series.drop(index=rated_cars, errors='ignore')
            # top n_recs highest predicted scores
            top_car_ids = scores_series.nlargest(n_recs).index.tolist()
            # map the car IDs back to car make model for collaborative recommendations
            collaborative_recs = self.car_data[self.car_data['carID'].isin(top_car_ids)]['Make Model'].tolist()

        # combine content and collaborative recommendations
        combined_recs = collaborative_recs + content_recs
        unique_recs = list(dict.fromkeys(combined_recs).keys()) # remove duplicates while keeping order
        return unique_recs[:n_recs]
        
if __name__ == "__main__":
    # get datasets
    cars_data_path = "./data/df_cars_clean.csv"
    ratings_data_path = "./data/df_ratings_clean.csv"
    cars_data = pd.read_csv(cars_data_path)
    ratings_data = pd.read_csv(ratings_data_path)
    #print("cars data: \n", cars_data)
    #print("ratings data: \n", ratings_data)

    # test
    car_make_model = 'jeep cherokee latitude'
    userID = 98305
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
    
