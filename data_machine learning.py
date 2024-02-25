import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import sklearn
import csv
import pickle
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR


##different machine learning model
# %matplotlib inline
# %matplotlib --list

# change path below to wherever you have stored the data
# change dataset below to work on second dataset
DATAPATH = r"C:\Users\Administrator\OneDrive - UGent\Desktop\qq's document\vacabulary_game\results2"
DATASET = 'output'

filename= os.path.join(DATAPATH, DATASET + '.csv')
df = pd.read_csv(filename)
df = df.fillna(0)

df.head()

Shape = df.values[:, 3:].shape  # 4144 samples, with 768 features.
print(Shape)
#
# features
X = df.values[:, 4:12]
X_all =  df.values[:, 4:13]
print(X_all)
# labels
y = df.values[:, -1].astype(np.float32)
# print(y)
#
# Split the data into training and test sets
test_size = 0.2  # Percentage of data to use for testing (adjust as needed)
random_state = 42  # Set a random seed for reproducibility (optional)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
train, test = train_test_split(X_all, test_size=test_size, random_state=random_state)
# print(train)
X_train = train[:, 0:8]
X_test = test[:, 0:8]
y_train = train[:, -1]
y_test = test[:, -1]

# compute min and max of each feature
X_train_min = np.min(X_train, axis=0)
X_train_max = np.max(X_train, axis=0)
vector = df.iloc[:, 4:12].columns.values
vector_all = df.iloc[:, 4:13].columns.values

print(vector)
print(X_train_min)
#
# # # plot min and max of each feature in a scatter plot
plt.figure(figsize=(24,24))
plt.scatter(vector, X_train_max, c="red")
plt.scatter(vector, X_train_min, c="blue")
plt.xlabel('Feature')
plt.ylabel('Values')
plt.show()
#
# scatter plot of the data
traindf = pd.DataFrame(train, columns=vector_all)
traindf.head(5)
plt.figure(figsize=(24,8))
for feature in vector:
    plt.scatter([feature]*X_train.shape[0], traindf[feature].values)
plt.xlabel('Feature')
plt.ylabel('Values')
plt.show()

# Normalize the data

scaler = RobustScaler()
scaler.fit(X_train)
X_train_pca = scaler.transform(X_train)
X_test_pca = scaler.transform(X_test)

# PCA on the data

pca = PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train_pca)
X_test_pca = pca.transform(X_test_pca)

# Plot PCA
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

## 4. Action required: Creating pipelines for preprocessing and feature selection

preprocessing = Pipeline([('robustscaler',RobustScaler())])
feature_selection = Pipeline([('selectkbest', SelectKBest())])  #  Feature selection. Do not modify!

### Linear, Lasso, and Ridge
param_grid = {
    'feature_selection__selectkbest__k': np.linspace(8, X_train.shape[1], 5, dtype=int),
    'classifier__fit_intercept': [True, False],
}

pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('feature_selection', feature_selection),
    ('classifier', LinearRegression())
    # ('classifier', Ridge())
    # ('classifier', Lasso())
    # ('classifier', SVR())

])
# # Ridge Regression, Lasso Regression, or Support Vector Regression.
# cv = KFold(n_splits=5, shuffle=True, random_state=42)
#
# gridsearch = GridSearchCV(pipeline, param_grid, n_jobs=2, cv=cv, verbose=1, return_train_score=True, error_score='raise')
# gridsearch.fit(X, y)
#
# results = gridsearch.cv_results_
# train_score = results['mean_train_score'][gridsearch.best_index_]
# test_score = results['mean_test_score'][gridsearch.best_index_]
#
# print('Training accuracy {}'.format(train_score))
# print('Test accuracy: {}'.format(test_score))
#
# print('Best estimator:')
# print(gridsearch.best_estimator_)
###

#SVR
param_grid = {
    'feature_selection__selectkbest__k': np.linspace(8, X_train.shape[1], 5, dtype=int),
    'classifier__C': [0.1, 1, 10],  # Adjust the C parameter values as needed
    'classifier__kernel': ['linear', 'rbf']  # Adjust the kernel type as needed
}

pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('feature_selection', feature_selection),
    ('classifier', SVR())
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

gridsearch = GridSearchCV(pipeline, param_grid, n_jobs=2, cv=cv, verbose=1, return_train_score=True, error_score='raise')
gridsearch.fit(X, y)

results = gridsearch.cv_results_
train_score = results['mean_train_score'][gridsearch.best_index_]
test_score = results['mean_test_score'][gridsearch.best_index_]

print('Training accuracy {}'.format(train_score))
print('Test accuracy: {}'.format(test_score))

print('Best estimator:')
print(gridsearch.best_estimator_)

#
fig = plt.figure(figsize = (5, 5))
ax = fig.add_subplot(projection='3d')

# Make data
### Linear, Lasso, and Ridge
# param_values = gridsearch.cv_results_['params']
# print(param_values)
#
# X = [1 if params['classifier__fit_intercept'] ==True else 0 for params in param_values]
# Y = gridsearch.cv_results_['param_feature_selection__selectkbest__k']
#
# #
# Z1 = results["mean_test_score"]
# Z2 = results["mean_train_score"]
# #
# # Plot the surface.
# ax.scatter(X, Y, Z1)
# ax.scatter(X, Y, Z2)
# plt.plot(results['param_feature_selection__selectkbest__k'].data, results['mean_test_score'])
# plt.plot(results['param_feature_selection__selectkbest__k'].data, results['mean_train_score'])
# plt.show()
####
# Make data.
### SVR
param_values = gridsearch.cv_results_['params']
# print(param_values)
#
X = gridsearch.cv_results_['param_feature_selection__selectkbest__k']
Y = gridsearch.cv_results_['mean_test_score']
Z = gridsearch.cv_results_['mean_train_score']

# Plot the surface.
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, Z)
ax.set_xlabel('SelectKBest K')
ax.set_ylabel('Mean Test Score')
ax.set_zlabel('Mean Train Score')
plt.show()
###

best_estimator = gridsearch.best_estimator_

best_estimator.fit(X_train, y_train)
prediction = best_estimator.predict(X_test)
print(prediction,y_test)
# y_test = np.random.rand(61)
mae = mean_absolute_error(prediction,y_test)
mse = mean_squared_error(prediction,y_test)
rmse = np.sqrt(mse)
r2 = r2_score(prediction,y_test)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

# Ridge
# Training accuracy 0.9823290541085525
# Test accuracy: 0.9758588281791495

#linear
# Training accuracy 0.9830055005435394
# Test accuracy: 0.972405678495068



#lasso
# Training accuracy 0.1020184212593026
# Test accuracy: 0.09142880975970782

#SVR
# Training accuracy 0.9767185670353381
# Test accuracy: 0.9736851917015606