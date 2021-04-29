#!/usr/bin/env python
# coding: utf-8


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.getcwd()
data = pd.read_csv("./Cali Hosuing Predict_files/housing.csv")



#exploring the dataset
data.info()


# In[10]:


#statistics about our dataset
#data.describe()
data.columns


#creating the feature matrix of selected features for prediciton
feature_names=['longitude','latitude','population','total_rooms',
              'housing_median_age','median_income']
X= data[feature_names]
X.head()




y = data.median_house_value
y[:2]



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import mean_absolute_error

X_train,X_test,y_train,y_test = train_test_split(X,y ,random_state = 0)

#define the model
model = DecisionTreeRegressor()
 #fit model
model.fit(X_train,y_train)
#predict
predictions = model.predict(X_test)
predictions[:2]
#we calculate the MAE to perform some model validation
print("MAE without tuning: {:,.0f}" . format(mean_absolute_error(y_test,predictions)))
print(model.score(X_test,y_test))



#we shall now perform some hyper-parameter tuning to find the optimal max_leaf_nodes that
#help model predict without risk of underfitting or overfitting.
def calculate_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train,y_train)
    preds_val = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return(mae)



#loop to find ideal tree size from list of several leaf_nodes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

scores = {leaf_size: calculate_mae(leaf_size,X_train,X_test, y_train, y_test) 
          for leaf_size in candidate_max_leaf_nodes}
#best value of leaf_nodes will be the one with minimum value of MAE
best_tree_size = min(scores, key=scores.get)

print(scores.values())



#using the best leaf_size, let's fit the model again
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size,random_state = 0)

# fit the final model and uncomment the next two lines
final_model.fit(X,y )

final_predict = final_model.predict(X)

print(mean_absolute_error(y,final_predict))
print(final_model.score(X,y))



final_predict[:10].round(1)

#final_model after tuning was worse than initial model.

#fitting this same model with a RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state =1)
forest_model.fit(X_train,y_train)
predictions = forest_model.predict(X_test)

print(mean_absolute_error(y_test,predictions))
# the score of 0.81 is far better when using a RandomForest than a Decision Tree..
print(forest_model.score(X_test, y_test))




