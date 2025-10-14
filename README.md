# Hybrid recommendation system

A hybrid recommendation system for cars.

Car data (CarsForSale): https://www.kaggle.com/datasets/chancev/carsforsale?resource=download

Car reviews data (Edmunds car review): https://www.kaggle.com/datasets/shreemunpranav/edmunds-car-review

## Exploratory Data Analysis (EDA) and Preprocessing of Data

Python notebook file for EDA and preprocessing

## 3 step hybrid 

Filter → Content-Based → Collaborative

### 1) Filter by Used / New, Dealer / Private

Apply a strict filter on Used/New and Dealer/Private.

### 2) Content-based recommendations 

Features we extract for content based recommendations:

**Year, Drivetrain, Fueltype, Transmission, ExteriorColor**

**Similarity / ranking:** Cosine similarity

### 3) Collaborative-filtering (CF) 
In this project, CF is implemented using Non-negative Matrix Factorization (NMF):
1. We build a user–item matrix where rows are users, columns are cars, and values are ratings.

2. NMF decomposes this matrix into latent features representing user preferences and item characteristics.

3. We predict ratings for unseen cars by combining user and item latent features.

This allows the system to recommend cars a user is likely to like based on patterns from similar users.

### Hybrid Recommendation
The hybrid recommender combines:

Content-based filtering (recommends similar cars based on features like make, model, drivetrain, and price)

Collaborative filtering (recommends cars based on user behavior)

The final recommendation list merges content-based and CF recommendations while removing duplicates.

This approach uses both car attributes and user preferences, making recommendations more accurate and personalized.

## Evaluation & tuning 