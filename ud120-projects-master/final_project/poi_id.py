# %load poi_id.py
#!/usr/bin/python

import math
import sys
import pickle
import numpy as np
from sklearn.grid_search import GridSearchCV
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','bonus', 'salary', 'percent_to_poi', 'percent_from_poi', 
                 'exercised_stock_options', 'total_payments', 'restricted_stock_deferred'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print len(data_dict)

for k,v in data_dict.items():
    print len(v)
    break
### Task 2: Remove outliers
data_dict.pop('TOTAL')

### Task 3: Create new feature(s)
### Creating 2 new features: percent_to_poi and percent_from_poi
### These features calculate the percentage of emails to and from an employee
### that were sent or received from a person of interest.
### They were calculated by taking the number of emails to or from a poi and dividing 
### it by the total number emails sent or received
for k,v in data_dict.items(): 
    if v['from_poi_to_this_person'] != 'NaN' and v['from_this_person_to_poi'] != 'NaN': 
        v['percent_from_poi'] = float(v['from_poi_to_this_person']) / float(v['to_messages'])
        v['percent_to_poi'] = float(v['from_this_person_to_poi']) / float(v['from_messages']) 
    else: 
        v['percent_from_poi'] = 'NaN' 
        v['percent_to_poi'] = 'NaN'

### Created a new feature: percent_stock_exercised
### this feature is the percent of stock an employee exercised given their total stock
for k,v in data_dict.items():
    if v['exercised_stock_options'] != 'NaN' and v['total_stock_value'] != 'NaN':
        v['percent_stock_exercised'] = float(v['exercised_stock_options']) / float(v['total_stock_value'])
    else:
        v['percent_stock_exercised'] = 'NaN'
        
count = 0
for k,v in data_dict.items():
    if v['poi'] == True:
        count += 1
        
print count


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
print "Fitting the classifier to the training set"
param_grid_kn = {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'p': [1,2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [5, 10, 20, 30, 50, 100, 200]
        }

param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [1, 5, 10, 15, 20, 40, 100],
        'min_samples_split': [2, 3, 4, 5]
        }

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
#clf = GridSearchCV(DecisionTreeClassifier(), param_grid_dt)
#clf = DecisionTreeClassifier(criterion = "entropy", max_depth=10)
#clf = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=20,
            #max_features=None, max_leaf_nodes=None,
            #min_impurity_decrease=0.0, min_impurity_split=None,
            #min_samples_leaf=1, min_samples_split=5,
            #min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            #splitter='best')

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
#clf = GridSearchCV(KNeighborsClassifier(), param_grid_kn)
clf = KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=1,
           weights='uniform')

# Random Forest
from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(criterion="entropy", max_depth=5, n_estimators=5)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.4, random_state=42)

clf = clf.fit(features_train, labels_train)
#print "Best estimator found by grid search:"
#print clf.best_estimator_
pred = clf.predict(features_test)


from sklearn.metrics import precision_score, recall_score

precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)

print clf.score(features_test,labels_test)
print precision
print recall

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)