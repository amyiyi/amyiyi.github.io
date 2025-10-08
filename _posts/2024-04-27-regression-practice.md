---
layout: post
title: "Regression Practice from Jupyter Notebook"
date: 2024-04-27
categories: machine-learning regression
---

# import library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# Create DataFrame
data = pd.read_csv('train.csv')
# Display the DataFrame
data.head()
# creating a backup copy of the data 
data_original = data.copy()
# Populating null Age values with the average age by sex, Pclass, 
data['Age'] = data.groupby(['Sex', 'Pclass'],group_keys=False)['Ag
# Plot before and after imputation
fig, axes = plt.subplots(1, 2, figsize=(6, 8), sharey=True)
sns.histplot(data_original['Age'], ax=axes[0], kde=True, color='re
sns.histplot(data['Age'], kde=True, ax=axes[1], color='green').set
plt.show()
PassengerId
Survived
Pclass
Name
Sex
Age
SibSp
Parch
0
1
0
3
Braund,
Mr. Owen
Harris
male
22.0
1
0
1
2
1
1
Cumings,
Mrs. John
Bradley
Florence
Briggs
Th...
female
38.0
1
0
2
3
1
3
Heikkinen,
Miss.
Laina
female
26.0
0
0
3
4
1
1
Futrelle,
Mrs.
Jacques
Heath
Lily May
Peel)
female
35.0
1
0
4
5
0
3
Allen, Mr.
William
Henry
male
35.0
0
0
In [2]:
In [3]:
In [4]:
Out[4]:
In [5]:
In [6]:
In [7]:


Explanation of Not Using The "Survived" Field
Explanation:
Using the Survived field for imputing the Age field can lead to data leakage.
Data leakage occurs when information from outside the training dataset is
used to create the model. This can lead to overly optimistic performance
estimates and poor generalization to new data.
Comments:
Using the Survived field to impute Age can introduce bias because the


survival status might be influenced by the age of the passengers, thereby
distorting the model's understanding of the relationship between age and
survival.
Chart Questions
What does the plt.subplots function do?
This is a new function you haven't seen before, how would you find out
more about it?
Regression Model
Explanation:
A regression model, such as logistic regression, can be used to predict a
binary outcome (like survival).
Step 1 Convert all categorical dimensions to numerical values.
Why?
Regression algorithms are based on mathematical operations and require
numerical input. Categorical variables, which represent qualitative data,
cannot be directly processed by these algorithms. Converting categorical
variables into numerical formats allows the model to interpret and analyze
relationships effectively.
# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_firs
data_original.Embarked.value_counts()
Embarked
S    644
C    168
Q     77
Name: count, dtype: int64
data.head()
In [8]:
In [9]:
Out[9]:
In [10]:


"Embarked_C" is the baseline and is represented when both dummy
columns are 0. For example:
If Embarked_Q and Embarked_S is 0, then the passenger embarked at
C.
If Embarked_Q is 1 and Embarked_S is 0, then the passenger embarked
at Q.
If Embarked_Q is 0 and Embarked_S is 1, then the passenger embarked
at S.
Practical Question
What did the pd.get_dummies function do?
Used when you need to convert categorical text data into numerical
format. Especially before feeding data into machine learning models.
How would you find out?
Step 2 Separate the data into features and target variables
Why
PassengerId
Survived
Pclass
Name
Age
SibSp
Parch
Ticke
0
1
0
3
Braund,
Mr. Owen
Harris
22.0
1
0
A/5 2117
1
2
1
1
Cumings,
Mrs. John
Bradley
Florence
Briggs
Th...
38.0
1
0
PC 1759
2
3
1
3
Heikkinen,
Miss.
Laina
26.0
0
0
STON/O
310128
3
4
1
1
Futrelle,
Mrs.
Jacques
Heath
Lily May
Peel)
35.0
1
0
11380
4
5
0
3
Allen, Mr.
William
Henry
35.0
0
0
37345
Out[10]:


Separating the data into features (input variables) and target variables
(output variable) clearly defines what the model needs to predict. Features
provide the information used to make predictions, while the target variable
is the outcome the model aims to predict.
# Define features we will use for the model
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', '
# define the target variable 
y = data['Survived']
Step 3 Separate the feature and target dimensions into train and test
Why
Separating data into training and testing sets allows us to validate the
model's performance. Training the model on one subset and testing it on
another helps assess how well the model generalizes to new, unseen data.
Training set: Used to teach the model.
Testing set: Used to evaluate how well the model performs on unseen
data.
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_siz
X_train : 90% of the feature data used to train the model.
X_test : 10% of the feature data used to test the model.
y_train : 90% of the target values Survived) for training.
y_test : 10% of the target values for testing.
test_size=0.1 : 10% of the data is reserved for testing.
If you donʼt set random_state , the split will be different each time you
run the code. If you do set it (like random_state=42 ), the split will be
reproducible.
Step 4 Training the logistic regression model
# Train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
A little bit of theory
In [11]:
In [12]:
In [13]:
Out[13]:
▾LogisticRegression
?
i
▸ Parameters


max_iter Parameter:
This parameter sets the maximum number of iterations that the optimization
algorithm can run to converge to the best solution. During each iteration, the
algorithm updates the coefficients slightly, moving towards the direction that
reduces the error.
If the algorithm converges (i.e., the changes in the coefficients become very
small) before reaching the maximum number of iterations, it stops early. If
the algorithm does not converge within the specified number of iterations, it
stops and may not have found the best solution. This can happen if the data
is complex or the learning rate is not well-tuned.
Practical Question
Why are we using a logistic regression model in this situation?
Because we want numerical output
Step 5 Use the trained model to predict the output
# Predict on the test set
y_pred = model.predict(X_test)
y_pred
array([0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 
0, 0, 0,
      1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 
0, 0, 0,
      1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 
0, 0, 1,
      0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 
0, 1, 1,
      0, 0])
Step 6 Compare the results of the predicted output with the actual answers
i.e. y_pred v y_test
# Evaluate the model using accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
Accuracy: 0.8555555555555555
Confusion Matrix:
[[46  8]
[ 5 31]]
In [14]:
In [15]:
Out[15]:
In [16]:


Technical Reminder
Confusion Matrix a tool typically used to evaluate the performance of
classification models, not regression models. It summarizes the number of
correct and incorrect predictions made by the model, comparing actual
target values with predicted values.
Accuracy Score a metric used to evaluate the correctness of predictions.
For classification, it is the ratio of the number of correct predictions to the
total number of predictions.
image: Research Gate 
Step 7 Calculate feature importance
Why
Calculating feature importance helps in understanding which features have
the most significant impact on the model's predictions. By identifying the
most important features, we can keep the most relevant features and
improve model performance.
# Calculate feature importance
feature_importance = model.coef_[0]
# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)
# Plot feature importance
plt.figure(figsize=(6, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()
In [20]:


Training = learning the relationship.
As you can see above, Fare do not have much impact on survival but who
have sex as male do.
Understanding Feature Importance Scores
Positive Importance Score: A positive coefficient indicates that as the
feature value increases, the likelihood of the positive class increases
(assuming binary logistic regression). In other words, higher values of this
feature are associated with a higher probability of the target being 1 (or the
positive class).
Negative Importance Score: A negative coefficient indicates that as the
feature value increases, the likelihood of the positive class decreases. This
means that higher values of this feature are associated with a higher
probability of the target being 0 (or the negative class).
Step 8 Transform the test data into the format required for the model
Import a completely new dataset called 'test.csv' to apply what the
trained model learn to new passengers whose survival is unknown.
Basically this step show how to take the model you trained (after
checking importance) and apply it to new, real-world passengers by
cleaning → transforming → predicting
# Import new test data
test_data = pd.read_csv('test.csv')
In [29]:


test_data.head()
# Populating null Age values with the average age by sex, Pclass, 
test_data['Age'] = test_data.groupby(['Sex', 'Pclass'],group_keys=
# check for null values 
test_data.isnull().sum()
PassengerId      0
Pclass           0
Name             0
Sex              0
Age              0
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
# using an average of sex and PClass for the missing fare value
test_data['Fare'] = test_data.groupby(['Sex', 'Pclass'],group_keys
# Preprocess the test data in the same way as the training data
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'],
# Ensure the test data has the same columns as the training data
test_data = test_data.reindex(columns=X.columns, fill_value=0)
# Predict on the new test data
PassengerId
Pclass
Name
Sex
Age
SibSp
Parch
Ticket
0
892
3
Kelly, Mr.
James
male
34.5
0
0
330911
1
893
3
Wilkes,
Mrs.
James
Ellen
Needs)
female
47.0
1
0
363272
2
894
2
Myles, Mr.
Thomas
Francis
male
62.0
0
0
240276
9
3
895
3
Wirz, Mr.
Albert
male
27.0
0
0
315154
8
4
896
3
Hirvonen,
Mrs.
Alexander
Helga E
Lindqvist)
female
22.0
1
1
3101298
12
In [30]:
Out[30]:
In [31]:
In [32]:
Out[32]:
In [33]:
In [34]:
In [35]:
In [36]:


test_predictions = model.predict(test_data)
# adding the survived field back to the test data
test_data['Survived_predicated'] = test_predictions
test_data['Survived_predicated'].value_counts() # try to add .plot
Survived_predicated
0    261
1    157
Name: count, dtype: int64
Practical Example
Create a new column FamilySize in the DataFrame, which is the sum of SibSp
and Parch.
Then, create a regression model to predict the Survived field using Pclass,
Age, FamilySize, Fare, Sex_male, Embarked_Q, and Embarked_S.
# -- Create a new column: FamilySize
data['FamilySize'] = data['SibSp'] + data['Parch']
# -- Separate the data into features and target variables
# define features we will use for the model
X = data[['Pclass','Age','FamilySize','Fare','Sex_male','Embarked_
# define the target variable 
y = data['Survived']
# -- Separate the feature and target dimensions into train and tes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_siz
# -- Train the logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
# -- Use the trained model to predict the output 
y_pred = model.predict(X_test)
y_pred
In [37]:
In [38]:
Out[38]:
In [39]:
In [40]:
In [41]:
In [42]:
Out[42]:
▾LogisticRegression
?
i
▸ Parameters
In [43]:
In [44]:


array([0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 
0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 
0, 0, 0,
      1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 
0, 0, 1,
      0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 
0, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 
0, 0, 0,
      1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 
0, 1, 0,
      0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 
0, 0, 1,
      0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 
1, 0, 0,
      0, 1, 1])
 
Out[44]:
In [45]:


