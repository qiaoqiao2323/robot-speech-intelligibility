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
y_user = y[:, 0]
y_correct = y[:, 1]
y = y[:, 0] + y[:, 1]  # Adjust the combination method as needed


preprocessing = Pipeline([('robustscaler',RobustScaler())])
feature_selection = Pipeline([('selectkbest', SelectKBest())])  #  Feature selection. Do not modify!

### Linear, Lasso, and Ridge
param_grid = {
    'feature_selection__selectkbest__k': np.linspace(9, X_train.shape[1], 5, dtype=int),
    'classifier__fit_intercept': [True, False],
}

pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('feature_selection', feature_selection),
    # ('classifier', LinearRegression())
    ('classifier', Ridge())
    # ('classifier', Lasso())
    # ('classifier', SVR())

])
# Ridge Regression, Lasso Regression, or Support Vector Regression.
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

y_test = y_test[:, 0] + y_test[:, 1]  # Adjust the combination method as needed

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
# Access the linear regression model within the pipeline
linear_regression_model = best_estimator.named_steps['classifier']

# Get the coefficients
coefficients = linear_regression_model.coef_
# Analyze predictor values (coefficients)
feature_names = ['volume' ,' speed', ' pitch', ' enuaciation']
# print(coefficients) [ 1.24595310e-01 -3.38983963e-02  3.16654439e-01  1.32869644e-02
#  -1.17808428e-03  1.27913777e-04  7.45196567e-03  2.09942056e-01]
coefficients_voice= coefficients[3:7]

# Plot coefficients
plt.figure(figsize=(10, 6))
plt.bar(feature_names, coefficients_voice)
plt.xlabel('Predictor Variables')
plt.ylabel('Coefficients')
plt.title('Linear Regression - Coefficients')
plt.xticks(rotation=90)
plt.show()


# Plot coefficients
plt.figure(figsize=(10, 6))
plt.bar(vector, coefficients)
plt.xlabel('Predictor Variables')
plt.ylabel('Coefficients')
plt.title('Linear Regression - Coefficients')
plt.xticks(rotation=90)
plt.show()

threshold = 11
below_threshold_indices = np.where(prediction < threshold)[0]
print(below_threshold_indices)
# while len(below_threshold_indices) > 0.5*len(below_threshold_indices):
# Threshold for modifying predictor values
# Modify predictor values based on coefficients
adjustment_value = [0.2,10,0.05,0.05]  # Define the adjustment value
# adjustment_value = 10000  # Define the adjustment value

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
# prediction = best_estimator.predict(X_train)
print(prediction_new)

# Re-fit the model with modified predictor values
# best_estimator.fit(X_train, y_train)
# y_test = y_test[:, 0] + y_test[:, 1]  # Adjust the combination method as needed
#
# print(prediction,y_test)
# # y_test = np.random.rand(61)
# mae = mean_absolute_error(prediction,y_test)
# mse = mean_squared_error(prediction,y_test)
# rmse = np.sqrt(mse)
# r2 = r2_score(prediction,y_test)
#
# # Evaluate the updated model
# prediction = best_estimator.predict(X_test)
# print(prediction)
#
# print("Mean Absolute Error (MAE):", mae)
# print("Mean Squared Error (MSE):", mse)
# print("Root Mean Squared Error (RMSE):", rmse)
# print("R-squared (R2) Score:", r2)


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
# Training accuracy 0.7767185670353381
# Test accuracy: 0.9736851917015606