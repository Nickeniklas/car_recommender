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

### 2) Content-based scoring 

Numeric: Price, Mileage, Age = current_year − Year, MPG = (MinMPG+MaxMPG)/2.

Ratings as features: ComfortRating, PerformanceRating, ReliabilityRating, etc.

Categorical: Make, Model, Transmission, FuelType, Drivetrain, colours. 

Scale numeric features (StandardScaler or MinMax).

**Similarity / ranking:** Use cosine similarity

### 3) Combine with ratings (collaborative)

Build an item quality score.

Normalize to [0,1].

Combine scores.

final_score = α * content_score + (1−α) * rating_score.

## Evaluation & tuning 