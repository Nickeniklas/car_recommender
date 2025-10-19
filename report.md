# Results

## Content based recommendations

**Output:**
```
Content-based recommendations:
                    Make Model Year     Score
carID
2100   gmc sierra 2500 denali 2018  1.000000
373         volkswagen tiguan 2012  0.939894
1278    jeep patriot latitude 2012  0.939894
1026             audi q5 2.0t 2012  0.939894
2252              audi q7 3.0 2010  0.856063
```

## Collaborative based recommendations
Collaborative recommendations:
          Make Model Year     Score
carID
1824   toyota camry 2018  1.000000
1705   toyota camry 2018  1.000000
2006   honda accord 2016  0.745574
261      ford f-150 2018  0.644229
1637     ford f-150 2018  0.644229
Hybrid recommendations:
                             Make Model Year     Score
carID
1824                      toyota camry 2018  0.500000
1705                      toyota camry 2018  0.500000
2100            gmc sierra 2500 denali 2018  0.500000
373                  volkswagen tiguan 2012  0.469947
1026                      audi q5 2.0t 2012  0.469947
1278             jeep patriot latitude 2012  0.469947
2252                       audi q7 3.0 2010  0.428032
2268      chevrolet silverado 1500 ltz 2021  0.422541
1940                         bmw 328 i 2012  0.420860
2300   mercedes-benz cls-class cls 550 2012  0.400030
## Hybrid 

**Output:**
```
Hybrid recommendations:
                             Make Model Year     Score
carID
1824                      toyota camry 2018  0.500000
1705                      toyota camry 2018  0.500000
2100            gmc sierra 2500 denali 2018  0.500000
373                  volkswagen tiguan 2012  0.469947
1026                      audi q5 2.0t 2012  0.469947
1278             jeep patriot latitude 2012  0.469947
2252                       audi q7 3.0 2010  0.428032
2268      chevrolet silverado 1500 ltz 2021  0.422541
1940                         bmw 328 i 2012  0.420860
2300   mercedes-benz cls-class cls 550 2012  0.400030
```

# How it went
### Datasets
For the two datasets to work with each other, they need matching carIDs. To accomplish this I made two main things:
```
def clean_model(model):
    """Remove common trim-levels from car model names."""
    # Remove common trim-levels (case insensitive)
    return re.sub(r'\b(se|le|xle|lx|ex|sport|limited|platinum|premium|plus|4MATIC|4matic|gt|v8|R/T|xl|sl|sr|base)\b', '', str(model).lower()).strip()
```
and then create a new "car_key" column which we match:
```
# recreate carkey for each Make-Model-Year combination (same as for df_cars) for both datasets
df_ratings['car_key'] = df_ratings['Make'] + "_" + df_ratings['Model'] + "_" + df_ratings['Year'].astype(str)
df_cars['car_key'] = df_cars['Make'] + "_" + df_cars['Model'] + "_" + df_cars['Year'].astype(str)
```
This gave me 26000+ ratings with a matching carID from the other dataset.  

### Content-based features
I found that using the cars Make and Model in the content-based features gave too many results of almost identical cars. *This could be solved in many ways, for example by cleaning the datasets even further or by coding the recommendations not to include same string as the seed item.* But I decided to remove the Make and Model from the features and got great results! 

After choosing the columns to use as features. Make one big 'Features' column with all features combined.
```
# combine string features into one column
df_cars['Features'] = (
    df_cars['Year'].astype(str) + ' ' + 
    df_cars['Drivetrain'] + ' ' +
    df_cars['FuelType'] + ' ' +
    df_cars['Transmission'] + ' ' +
    df_cars['ExteriorColor']
)
```
So this is also a list of features used in the content-based recommendation.

### Collaborative filtering
After preprocessing the dataset, collaborative filtering was surprisingly easy to implement! 

### Hybrid 
Combine collaborative and content-based with formula:

Sₕᵧbᵣᵢd(i, u) = α · S_CF(i, u) + (1 − α) · S_CB(i)

Where:

Sₕᵧbᵣᵢd(i, u) → final hybrid score for item i and user u

S_CF(i, u) → collaborative filtering predicted rating

S_CB(i) → content-based similarity score

α ∈ [0, 1] → weight controlling the balance between CF and CB

Interpretation:

α = 1 → fully collaborative

α = 0 → fully content-based

0 < α < 1 → balanced hybrid

#### Return value
In the beginning I returned a list of Make and Model to easily print human readably lists. But I found that everything works the best with each other using series of carIDs. So I kept everything as Pandas series', and instead made a method inside HybridRecommender; def id_to_title() which converts the IDs into clean lists with make, model, year and score (New better output seen above).

Old output (list of car titles):
```
Hybrid Recommendations for user 98305 and car jeep cherokee latitude:
['ford f-150', 'honda cr-v', 'toyota tundra', 'honda accord', 'ram 1500']
```

### Evaluator
Class Evaluator() with methods for evaluating. Since coverage@k() and novelty() need all recommendations for all users, get_all_recs() method was created.

#### precision@k
Fraction of top‑k recommended items that are actually relevant for the user.

Usually precision@k will always be 0, Since we exclude items that the user has already rated. But if commenting the lines out, results:
```
Precision@k:
 0.058823529411764705
```

#### coverage@k
Fraction of all items in the catalog that appear in any users top‑k recommendations.

only look at top-k items per user, not every single item the system could possibly recommend. 
Output:
```
Coverage@k:
 0.08262994224788983
```
Which means ~8.3% is covered.

#### calculate_novelty
how uncommon or “surprising” the recommended items are, based on how few users know/rated them.

Output:
```
Mean Novelty of all users: 7.3708766002720525
```
This value is expected to be quite high in our case, with the amount of cars compared to ratings.

### Setup
I played around with different setups, until I came to the conclusion that having different classes works best. Now content, collaborative can be used independantly and is easier to debug. Also allows the main block to be minimal and clean. After that implementing evaluation was simple. The Evaluator() class has differents methods for evaluating, and can easily be modified or developed further. Implementing worked well since all other classes return pandas series'.

___
## Sources
Assisted by ChatGPT, Copilot.

### Data
Car data (CarsForSale): https://www.kaggle.com/datasets/chancev/carsforsale?resource=download

Car reviews data (Edmunds car review): https://www.kaggle.com/datasets/shreemunpranav/edmunds-car-review

### Other
Recommenders on GeeksforGeeks: https://www.geeksforgeeks.org/machine-learning/what-are-recommender-systems/

Decision support systems on GeeksforGeeks: https://www.geeksforgeeks.org/business-studies/decision-support-system/

