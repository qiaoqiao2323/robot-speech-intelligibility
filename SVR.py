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

Shape = df.values[:, 4:13].shape  # 4144 samples, with 768 features.
print(Shape)
#
# features
X = df.values[:, 4:13]
X_all =  df.values[:, 4:14]
print(X_all)
# labels
y = df.values[:, 12:14].astype(np.float32)
print(y)

# Split the data into training and test sets
test_size = 0.2  # Percentage of data to use for testing (adjust as needed)
random_state = 42  # Set a random seed for reproducibility (optional)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
train, test = train_test_split(X_all, test_size=test_size, random_state=random_state)
# print(train)
X_train = train[:, 0:8]
X_test = test[:, 0:8]
y_train = train[:, 7:9]
y_test = test[:, 7:9]

# compute min and max of each feature
X_train_min = np.min(X_train, axis=0)
X_train_max = np.max(X_train, axis=0)
vector = df.iloc[:, 4:12].columns.values
vector_all = df.iloc[:, 4:14].columns.values

print(vector)
print(X_train_min)


# # # plot min and max of each feature in a scatter plot
plt.figure(figsize=(24,24))
plt.scatter(vector, X_train_max, c="red")
plt.scatter(vector, X_train_min, c="blue")
plt.xlabel('Feature')
plt.ylabel('Values')
# plt.show()

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

y_train = y_train[:, 0] + y_train[:, 1]  # Adjust the combination method as needed

# Plot PCA
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.show()

## 4. Action required: Creating pipelines for preprocessing and feature selection

y = y[:, 0] + y[:, 1]  # Adjust the combination method as needed

preprocessing = Pipeline([('robustscaler',RobustScaler())])
feature_selection = Pipeline([('selectkbest', SelectKBest())])  #  Feature selection. Do not modify!


# SVR
param_grid = {
    'feature_selection__selectkbest__k':  [X_train.shape[1]],  # Include all features
    'classifier__C': [0.1, 1, 10],  # Adjust the C parameter values as needed
    # 'classifier__kernel': [ 'rbf']  # Adjust the kernel type as needed
    # 'classifier__kernel': ['linear']  # Adjust the kernel type as needed
    # 'classifier__kernel': ['linear','rbf']  # Adjust the kernel type as needed
    # Creating an SVR model with a polynomial kernel of degree 3
    # 'classifier__kernel': ['sigmoid']
    'classifier__kernel': ['poly'], # Adjust the kernel type as needed
    'classifier__degree': [3]  # Desired degree values to be tested

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

# Make data.
### SVR
# param_values = gridsearch.cv_results_['params']
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
# plt.show()

###
best_estimator = gridsearch.best_estimator_

best_estimator.fit(X_train, y_train)
prediction = best_estimator.predict(X_test)
####Predicition
best_estimator.fit(X_train, y_train)

y_test = y_test[:, 0] * y_test[:, 1]  # Adjust the combination method as needed

print(prediction,y_test)
# y_test = np.random.rand(61)
mae = mean_absolute_error(prediction,y_test)
mse = mean_squared_error(prediction,y_test)
rmse = np.sqrt(mse)
r2 = r2_score(prediction,y_test)

# Evaluate the updated model
prediction = best_estimator.predict(X_test)
print(prediction)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)


# Find a way
# Access the SVR model within the pipeline
svr_model = best_estimator.named_steps['classifier']


# dual_coefficients array has a length of 20,
# it means that you have 20 support vectors in your SVR model.
# The number of support vectors does not necessarily match the number of input features.
# Each support vector represents a point in the input space,
# and its corresponding dual coefficient indicates its importance in the prediction.

# To relate the dual_coefficients to your 8 input features,
# you need to consider the support vectors and their associated indices.
# Here's an approach to associate the coefficients with the features
support_vectors = svr_model.support_vectors_
support_vector_indices = svr_model.support_
absolute_dual_coefficients = np.abs(svr_model.dual_coef_)
dual_coefficients = svr_model.dual_coef_

# The dual coefficients indicate the importance or influence
# of each support vector in the SVM model.
# The absolute values of these coefficients provide
# a measure of the magnitude or significance of each
# support vector's contribution to the model.

feature_importance = np.mean(absolute_dual_coefficients, axis=0)
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_importance = feature_importance[sorted_indices]

# Print the associated features
# print(dual_coefficients)
# print(support_vector_indices)
# print("Support Vector:", support_vectors)
# print(sorted_importance)
# print(vector)

# for index in  range(len(sorted_indices)):
# for feature,suvector in zip(vector,support_vectors):
#     # support_vector_dict = {feature: value for feature, value in zip(vector, support_vectors)}
#     # support_vector_dict = {feature: value for value in suvector}
#     support_vector_dict = {vector[i]: value for i, value in enumerate(suvector)}
#     print(support_vector_dict)
# #
#
# support_vector_dict = {}
# for each in range(support_vectors.shape[0]):
#     row = support_vectors[each]
#     row_dict = {}
#     for feature, value in zip(vector, row):
#         row_dict[feature] = value
#     print('row_dict: ', row_dict)
#     support_vector_dict[str(each)] = row_dict
# print('support_vector_dict: \n', support_vector_dict)


support_vector_dict = {}
for fea_index, feature in enumerate(vector):
    support_vector_dict[feature] = support_vectors[:, fea_index]
    featuresss = support_vectors[:, fea_index]
# print(support_vector_dict)
my_list = list(support_vector_dict.values())
x_values = range(len(support_vector_dict))
y_values = list(support_vector_dict.values())
sum_list = [np.sum(arr) for arr in support_vector_dict.values()]
# print(sum_list)

# Modify predictor values based on coefficients
adjustment_value = [0.2,10,0.05,0.05]  # Define the adjustment value
# adjustment_value = 10000  # Define the adjustment value
feature_names = ['volume' ,' speed', ' pitch', ' enuaciation']
# print(coefficients) [ 1.24595310e-01 -3.38983963e-02  3.16654439e-01  1.32869644e-02
#  -1.17808428e-03  1.27913777e-04  7.45196567e-03  2.09942056e-01]
coefficients_voice= sum_list[3:7]
threshold = 10
below_threshold_indices = np.where(prediction < threshold)[0]

for index in below_threshold_indices:
    for adindex, (feature, coefficient) in enumerate(zip(feature_names, coefficients_voice)):
        if coefficient < 0:
            # Increase or decrease the predictor values based on their relationship with the predicted values
            X_test[index, vector == feature] -= adjustment_value[adindex]
        if coefficient > 0:
            # Increase or decrease the predictor values based on their relationship with the predicted values
            X_test[index, vector == feature] += adjustment_value[adindex]

prediction_new = best_estimator.predict(X_train)
below_threshold_indices = np.where(prediction < threshold)[0]
print('x_modified = ', X_test)

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