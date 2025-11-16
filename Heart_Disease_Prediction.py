Importing the Dependencies
"""

pip install lazypredict

pip install pip install scikit-learn==1.5.1

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lazypredict
from lazypredict.Supervised import LazyClassifier
import sklearn

"""Data Collection and Processing"""

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('/content/heart_disease_data.csv')

# print first 5 rows of the dataset
heart_data.head()

# print last 5 rows of the dataset
heart_data.tail()

# number of rows and columns in the dataset
heart_data.shape

# getting some info about the data
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of Target Variable
heart_data['target'].value_counts()

"""1 --> Defective Heart

0 --> Healthy Heart

**Normalization of Data**
"""

from sklearn.preprocessing import MinMaxScaler
standardScaler = MinMaxScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak','sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
heart_data[columns_to_scale] = standardScaler.fit_transform(heart_data[columns_to_scale])

heart_data.head()

"""Splitting the Features and Target"""

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)

print(Y)

"""Splitting the Data into Training data & Test Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=13)

print(X.shape, X_train.shape, X_test.shape)

"""**Prediction of Accuracy Level of Different Classifiers**"""

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, Y_train, Y_test)

print(models)

"""**Model Training**

Logistic Regression
"""

model = RandomForestClassifier()

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

"""# Model Evaluation

Accuracy Score
"""

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

"""Saving Model"""

import pickle

pickle.dump(model,open('/content/Heart_Disease_Trained_Model_151','wb'))

"""Loading Model"""

Trained_Model_Loaded=pickle.load(open('/content/Heart_Disease_Trained_Model_151','rb'))

"""Testing"""

input_data = (60,1,0,125,258,0,0,141,1,2.8,1,1,3)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = Trained_Model_Loaded.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
