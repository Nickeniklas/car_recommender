# Hybrid recommendation system

A hybrid recommendation system for cars.

Dataset: https://www.kaggle.com/datasets/chancev/carsforsale?resource=download

## Exploratory Data Analysis (EDA)

## 3 step hybrid 
Filter → content ranking → rating adjustment 

### 1) Filter by Used / New

Apply a strict filter on Used/New.

### 2) Content-based scoring (rank the candidates)

Numeric: Price, Mileage, Age = current_year − Year, MPG = (MinMPG+MaxMPG)/2.

Ratings as features: ComfortRating, PerformanceRating, ReliabilityRating, etc.

Categorical: Make, Model, Transmission, FuelType, Drivetrain, colours. 

Scale numeric features (StandardScaler or MinMax).

**Similarity / ranking:** Use cosine similarity

### 3) Combine with ratings (collaborative-style signal)

Build an item quality score.

If you have counts: compute a Bayesian average to avoid small-sample bias:
rating_score = (v/(v+m))*R + (m/(v+m))*C
where v = #reviews, R = item avg rating, C = global avg rating, m = smoothing constant (e.g., 5–10).

If no counts, use raw ConsumerRating but add conservative smoothing.

Normalize rating_score to [0,1].

Combine scores.

final_score = α * content_score + (1−α) * rating_score.

Start with α ≈ 0.7 (favor content). Tune later.
Output: final ranked list.

## Evaluation & tuning 