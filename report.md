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
Hybrid Recommendations for user 98305 and car jeep cherokee latitude:
['ford f-150', 'honda cr-v', 'toyota tundra', 'honda accord', 'ram 1500']
```

# How it went
### Content-based features
I found that using the cars Make and Model in the content-based features gave too many results of almost identical cars. *This could be solved in many ways, for example by cleaning the datasets even further or by coding the recommendations not to include same string as the seed item.* But I decided to remove the Make and Model from the features and got great results! 

### Collaborative filtering
After preprocessing the dataset, collaborative filtering was surprisingly easy to implement! 

### Setup
I played around with different setups, until I came to the conclusion that having different classes works best. Now content, collaborative can be used independantly and is easier to debug. Also allows the main block to be minimal and clean. After that implementing evaluation was simple. The Evaluator() class has differents methods for evaluating, and can easily be modified or developed further.  

In the beginning I returned a list of Make and Model to easily print human readably lists. But I found that everything works the best with each other using series of carIDs. So I kept everything as Pandas series', and instead made a method inside HybridRecommender; def id_to_title() which converts the IDs into clean lists with make, model, year and score.
