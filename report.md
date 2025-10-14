# Results

## Content based recommendations

**Output:**
```
Content based Recommendations for jeep cherokee latitude:
['ram 1500', 'jeep cherokee', 'ram 1500 laramie', 'jeep grand cherokee summit', 'ram 1500 rebel']
```

## Collaborative based recommendations

## Hybrid 

**Output:**
```
Hybrid Recommendations for user 98305 and car jeep cherokee latitude:
['ford f-150', 'honda cr-v', 'toyota tundra', 'honda accord', 'ram 1500']
```

# Findings
## Content-based filtering features
I found that using the cars Make and Model in the content-based filtering features gave too many results of almost identical cars. *This could be solved in many ways, for example by cleaning the datasets even further or by coding the recommendations not to include same string as the seed item.* But I decided to remove the Make and Model from the features and got great results! 
