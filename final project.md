
# Identify Fraud from Enron Email

### Author: Francisco Cerezo
****




## 1. Understanding the Dataset and Question
***

## Question 1:
___Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]___



The Enron fraud case, publicized in October 2001, eventually led to the bankruptcy of the Enron Corporation, an American energy company based in Houston, Texas, and the de facto dissolution of Arthur Andersen, which was one of the five largest audit and accountancy partnerships in the world.
At the end of the investigation the email database was put on the public domain to be used for historical research and academic purposes.

The purpose of this project is to use machine learning techniques to identify person of interest (POIS) within the fraud case  based of this email database. 
POI's definition is people: indicted, settled without admitting guilt, or testified in exchange of immunity.

### Dataset exploration

The data about the Enron email and financial data has beenn preoproceseded into a dictionary, where each key-value pair in the dictionary corresponds to one person. Exploring this dictionary we can observe the following characteristics:


```python
import pickle
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
```


```python
print "Number of elements in the dataset: "+str(len(enron_data))
```

    Number of elements in the dataset: 146
    


```python
i=0
for elem in enron_data:
    if enron_data[elem]["poi"]==1:
        i=i+1
    
print "Number of POIs: "+ str(i)
print "Number of non-POIs: "+ str(len(enron_data)-i)
```

    Number of POIs: 18
    Number of non-POIs: 128
    

As we can see there are only 18 from 146 people categorized as POI in our dataser. This means that our dataset is clearly unbalanced on POI category and we shoul taki this into consideration when selecting, training and tuning up our clasiffier.

Note:In the "poi_names.txt" we have  35 POIs manually identify. We could thin in to complete de dataset with this additional POIs. Unfortanately we do not have the finantial information related to ths additional POIs.


```python
print "Number of Features used: "+str(len(enron_data['METTS MARK'].keys()))
```

    Number of Features used: 21
    


```python
pprint.pprint(enron_data[next(iter(enron_data))])
```

    {'bonus': 600000,
     'deferral_payments': 'NaN',
     'deferred_income': 'NaN',
     'director_fees': 'NaN',
     'email_address': 'mark.metts@enron.com',
     'exercised_stock_options': 'NaN',
     'expenses': 94299,
     'from_messages': 29,
     'from_poi_to_this_person': 38,
     'from_this_person_to_poi': 1,
     'loan_advances': 'NaN',
     'long_term_incentive': 'NaN',
     'other': 1740,
     'poi': False,
     'restricted_stock': 585062,
     'restricted_stock_deferred': 'NaN',
     'salary': 365788,
     'shared_receipt_with_poi': 702,
     'to_messages': 807,
     'total_payments': 1061827,
     'total_stock_value': 585062}
    

Attending to this the list of features for each people incuded in the dataset  can be organized in 3 different gorups:

__financial features: __['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

__email features: __['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

__POI label: __[‘poi’] (boolean, represented as integer)



```python
missing_values = {}
for feature in enron_data['METTS MARK'].keys():
    missing_values[feature] = 0

for elem in data_dict:
    for f in data_dict[elem]:
        if data_dict[elem][f] == 'NaN':
            missing_values[f] += 1
            # fill NaN values
            #data_dict[emp][f] = 0

pprint.pprint(missing_values) 
```

    {'bonus': 63,
     'deferral_payments': 105,
     'deferred_income': 95,
     'director_fees': 127,
     'email_address': 33,
     'exercised_stock_options': 43,
     'expenses': 50,
     'from_messages': 58,
     'from_poi_to_this_person': 58,
     'from_this_person_to_poi': 58,
     'loan_advances': 140,
     'long_term_incentive': 79,
     'other': 53,
     'poi': 0,
     'restricted_stock': 35,
     'restricted_stock_deferred': 126,
     'salary': 50,
     'shared_receipt_with_poi': 58,
     'to_messages': 58,
     'total_payments': 21,
     'total_stock_value': 19}
    

We can see that there is a lot of missing values in the features of the dataset. This is another inportant point that we should keep in ming when doing the features selection.


After this quick dataset exploration, we know that he whole dataset is composed by 146 peoople and only 18 of them are POIs. On adittiion to it, there are a lot of values mission in some of the features.


The low size of the dataset, the unbalancing in POIs categorie and the existence of missing values are very important points that we should take into consideration later on, when we work on the feature selection and also in hte clasifier selection, and tune up. 
Due to all this problems, probably the performance of the classfier will be not as good as we would like it. Anyway we will see what can we get with this dataset.

### Outliers

As we discovered previously in during the miniproject, if we plot bonus vs. salary, there is an outlier datapoint representing the 'TOTAL' due to a spreadsheet quirk. 


```python
#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
```


![png](output_17_0.png)


If we remove the outlier as ploy the chart agais we can see de diference now:


```python
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
```


![png](output_19_0.png)


By Inspenting the list of people included on the dataset I realized about another strange value 'THE TRAVEL AGENCY IN THE PARK' tha clearly not represent to a separated person so idecided to removed also.


```python
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
```




    {'bonus': 'NaN',
     'deferral_payments': 'NaN',
     'deferred_income': 'NaN',
     'director_fees': 'NaN',
     'email_address': 'NaN',
     'exercised_stock_options': 'NaN',
     'expenses': 'NaN',
     'from_messages': 'NaN',
     'from_poi_to_this_person': 'NaN',
     'from_this_person_to_poi': 'NaN',
     'loan_advances': 'NaN',
     'long_term_incentive': 'NaN',
     'other': 362096,
     'poi': False,
     'restricted_stock': 'NaN',
     'restricted_stock_deferred': 'NaN',
     'salary': 'NaN',
     'shared_receipt_with_poi': 'NaN',
     'to_messages': 'NaN',
     'total_payments': 362096,
     'total_stock_value': 'NaN'}





## Question 2:
___ What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]___


### Create new features 

I decided to engineer two new features and add them into my dataset:
    
__fraction_from_poi:__   ratio of the messages from POI to this person against all the messages sent to this person.

__fraction_to_poi:__ ratio from this person to POI against all messages from this person   

The assumption wich drives me to create those features is that the percentaje of mails that a POI sent/receives to/from a other OIS should be higher than the non-POIs.


```python
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

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    data_point["fraction_from_poi"] = computeFraction( from_poi_to_this_person, to_messages )

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    data_point["fraction_to_poi"] = computeFraction( from_this_person_to_poi, from_messages )


```

    {'bonus': 600000,
     'deferral_payments': 'NaN',
     'deferred_income': 'NaN',
     'director_fees': 'NaN',
     'email_address': 'mark.metts@enron.com',
     'exercised_stock_options': 'NaN',
     'expenses': 94299,
     'fraction_from_poi': 0.04708798017348203,
     'fraction_to_poi': 0.034482758620689655,
     'from_messages': 29,
     'from_poi_to_this_person': 38,
     'from_this_person_to_poi': 1,
     'loan_advances': 'NaN',
     'long_term_incentive': 'NaN',
     'other': 1740,
     'poi': False,
     'restricted_stock': 585062,
     'restricted_stock_deferred': 'NaN',
     'salary': 365788,
     'shared_receipt_with_poi': 702,
     'to_messages': 807,
     'total_payments': 1061827,
     'total_stock_value': 585062}
    

### Intelligently select features  an Feature scaling
In order to select the features idecided to use the SelectKBest function, which selects the K features with the highrst scores. I obtained the following results on my dataset:

```python
# Features
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','fraction_from_poi',
 'fraction_to_poi']
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
POI_label = ['poi']

my_dataset=data_dict
my_features=POI_label + financial_features + email_features
my_features.remove('email_address')

## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# K-best features
k_best = SelectKBest(k=10)
k_best.fit(features, labels)

results_list = zip(k_best.get_support(),my_features[1:], k_best.scores_)
results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
pprint.pprint (results_list)

```

    [(True, 'exercised_stock_options', 24.815079733218194),
     (True, 'total_stock_value', 24.182898678566872),
     (True, 'bonus', 20.792252047181538),
     (True, 'salary', 18.289684043404513),
     (True, 'fraction_to_poi', 16.409712548035792),
     (True, 'deferred_income', 11.458476579280697),
     (True, 'long_term_incentive', 9.9221860131898385),
     (True, 'restricted_stock', 9.212810621977086),
     (True, 'total_payments', 8.7727777300916809),
     (True, 'shared_receipt_with_poi', 8.5894207316823774),
     (False, 'loan_advances', 7.1840556582887247),
     (False, 'expenses', 6.0941733106389666),
     (False, 'from_poi_to_this_person', 5.2434497133749574),
     (False, 'other', 4.1874775069953785),
     (False, 'fraction_from_poi', 3.128091748156737),
     (False, 'from_this_person_to_poi', 2.3826121082276743),
     (False, 'director_fees', 2.126327802007705),
     (False, 'to_messages', 1.6463411294420094),
     (False, 'deferral_payments', 0.22461127473600509),
     (False, 'from_messages', 0.16970094762175436),
     (False, 'restricted_stock_deferred', 0.065499652909891237)]
    

Before use SelectKBest function i performed some feature scaling since the  magnitures are quite different and it could impact in our analysis.
Finally I decided to use K=10 so I finally got the 10 top ranked features: 'exercised_stock_options','total_stock_value','bonus','salary''fraction_to_poi' , 'deferred_income', 'long_term_incentive', 'restricted_stock' 'total_payments','shared_receipt_with_poi'
In order to complement my ferture selection I crated a decision tree classfier and  I used "_feature_importance_" to ranked thoe more importante features for the clasiffier
I merged both results Kscore and _feature_importance and I decided to select the following features: 


```python
selected_features=['exercised_stock_options','total_stock_value','bonus','salary', 'fraction_to_poi','deferred_income','long_term_incentive','restricted_stock','total_payments','shared_receipt_with_poi']
```



### ¿Question 3?
___ What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]___

I decided to compare 3 classifiers: Naive Bayes, decision tree and random forest and i tried the performance using the "tester.py" function provided by udacity as helper:



```python
# Create Naive Bayes calssifier
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()

# Create decision tree calssifier with defaults pparameters
clf2 = tree.DecisionTreeClassifier()

# Create Ramdom Forest calssifier 
from sklearn.ensemble import RandomForestClassifier
clf3= RandomForestClassifier(max_depth=2, random_state=0)

#use tester.py to compare accuracy, precision, recall F1
test_classifier(clf1, my_dataset, features_list)
test_classifier(clf2, my_dataset, features_list)
test_classifier(clf3, my_dataset, features_list)

```

![alt text](performance.png "Results")

Based on this results I selected the decision tree classifier which maximices Prcision,Recall and F1 and F2 and had a similar accuracy than the rest.

### Question 4:
___What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]___

The most of the clasifier owns a set of parameter tha can be modified to improve the results over our dataset. The Default values for the classifier are not always the ones with the better performance, so you need to try different paramaters configuration in order to find the best combination.

I used "GridSearchCV" to try several parameters within my Decision tree classifier:


```python
from sklearn import grid_search
from sklearn.cross_validation import StratifiedShuffleSplit
# Create the parameter set to be tested in our Decision Tree clasifier
parameters = {'criterion':('gini', 'entropy'),
              'min_samples_split':[2,3,4],
              'max_depth':[None,2,4,6]
              }

# Get features & labels
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# Use StratifiedShuffleSplit to avoid problems dur to the higly balance & small dataset
cv = StratifiedShuffleSplit(labels, n_iter = 1000,random_state = 42)
# Run test over different parameters
clfs = grid_search.GridSearchCV( tree.DecisionTreeClassifier(), param_grid = parameters,cv = cv, scoring = 'f1').fit(features, labels)
# Get best estimator from results
clf= clfs.best_estimator_
print  clf

#test the final performance of our tune Decision tree clasiffier
test_classifier(clf, my_dataset, features_list)
```

As it is shown in the code the parameters and the values that I select to test were:

- Criterion:['gini', 'entropy'] The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.

- min_samples_split':[2,3,4] The minimum number of samples required to split an internal node

- max_depth':[None,2,4,6] The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.






### ¿Question 5?
What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]



### ¿Question 6?
Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

