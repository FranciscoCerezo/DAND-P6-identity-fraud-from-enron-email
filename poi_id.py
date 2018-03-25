#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import pprint
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','fraction_from_poi',
 'fraction_to_poi']
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
POI_label = ['poi']
features_list = POI_label + financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit

# Extract data to be plot in the scatter chart
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
# plot the scatter chart
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### Remove 'TOTAL' outlier and plot again
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

# plot the scatter chart without outlier
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### Remove 'THE TRAVEL AGENCY IN THE PARK' element
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


### Task 3: Create new feature(s)
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction=0
    if poi_messages != 'NaN' and all_messages != 'NaN':
		fraction = poi_messages/float(all_messages)
    return fraction

for name in data_dict:

    data_point = data_dict[name]
    # add fraction_from_poi to the dataset as ratio of the messages from POI to this person
    # against all the messages sent to this person.
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    data_point["fraction_from_poi"] = computeFraction( from_poi_to_this_person, to_messages )
    # add fraction_to_poi to the dataset as ratio from this person to POI against all messages
    #from this person
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    data_point["fraction_to_poi"] = computeFraction( from_this_person_to_poi, from_messages )


### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list.remove('email_address')


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
# Note that we do it just to show feature scaling example. If fact the final classifier algortihm chosen(decission tree)
# is not based in Euclidean distances and it is not affected for the different magnitudes in the features.





#######################################################
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(labels, n_iter = 1000,random_state = 42)
kbest = SelectKBest(f_classif)
Kbest_dict={'kbest__k': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]}
clf_list=[GaussianNB(),tree.DecisionTreeClassifier(), RandomForestClassifier(max_depth=2, random_state=0)]

# Visualize the impact of the number of feture in the F1score
#for elem in clf_list:
#    pipeline = Pipeline([('kbest', kbest), ('clasif', elem)])
#    grid_search = GridSearchCV(pipeline,Kbest_dict ,cv=cv, scoring = 'f1')
#    grid_search.fit(features,labels)
#    pprint.pprint(grid_search.grid_scores_)


# use K-best to rank the best features
k_best = SelectKBest()
k_best.fit(features, labels)
results_list = zip(k_best.get_support(),features_list[1:], k_best.scores_)
results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
# print the scores for each feature
pprint.pprint (results_list)

# use feature_importances_ from a decision tree classifier to rank the best features
from tester import test_classifier, dump_classifier_and_data
from sklearn import tree
clf_test = tree.DecisionTreeClassifier()
test_classifier(clf_test, my_dataset, features_list)
importance=clf_test.feature_importances_
for i in range (len(importance)):
    print features_list[i+1] + ": "+str(importance[i])
    
# Using the ranking from K-best anf merging it with the more important features obtained from de Decision tree clasifier
# I decided to select the best of both sets:   
features_list =['poi','exercised_stock_options','total_stock_value','salary', 'fraction_to_poi', 'restricted_stock','shared_receipt_with_poi']

    
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# Create Naibe Bayes classifier
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()

# Create DecisionTree classifier with defaults pparameters
clf2 = tree.DecisionTreeClassifier()

# Create Ramdom Forest classifier 
from sklearn.ensemble import RandomForestClassifier
clf3= RandomForestClassifier(max_depth=2, random_state=0)

# use tester.py to compare accuracy, precision, recall F1
test_classifier(clf1, my_dataset, features_list)
test_classifier(clf2, my_dataset, features_list)
test_classifier(clf3, my_dataset, features_list)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Create the parameter set to be tested in our Decision Tree clasifier
parameters = {'criterion':('gini', 'entropy'),
              'min_samples_split':[2,3,4],
              'max_depth':[None,2,4,6]
              }

# Get features & labels
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# Use StratifiedShuffleSplit to avoid problems due to the higly balanced cateogories ans the & small size of our  dataset
cv = StratifiedShuffleSplit(labels, n_iter = 1000,random_state = 42)
# Run test over the different parameters
clfs = GridSearchCV(tree.DecisionTreeClassifier(), param_grid = parameters,cv = cv, scoring = 'f1').fit(features, labels)
# Get best estimator from results
clf= clfs.best_estimator_



#test the final performance of our tune Decision tree classifier
test_classifier(clf, my_dataset, features_list)

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
# train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)