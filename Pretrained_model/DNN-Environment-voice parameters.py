from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
tf.keras.initializers.glorot_uniform
tf.keras.initializers.Zeros
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
from keras.constraints import non_neg
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
pd.set_option('display.max_rows', None)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import load_model


# fix random seed for reproducibility
seed = 2022
np.random.seed(seed)
tf.random.set_seed(seed)


filename = r"C:\Users\Administrator\PycharmProjects\data_processing_word\DATA_pre\Envoice.csv"
df = pd.read_csv(filename)

# Split the data into features and labels
X = df.iloc[:, 1:6].values  # Features
y = df.iloc[:, 6:10].values  # Labels
# Define your predictors (input) and targets (output) data
# Make sure to replace X_train and y_train with your actual data
print("X:", X)
print("y:", y)

# Split the data into training and test sets
test_size = 0.2
random_state = 42
X_train, X_test_original, y_train, y_test_original = train_test_split(X, y, test_size=test_size, random_state=random_state)
# 生成四个不同的 weight1 值
weight1_values = [0.5,0.2,0.4]

# 存储生成的数据的列表
X_weighted_list = []
y_weighted_list = []

# 假设你的数据是 X 和 y
# 切分数据成两部分
X_half1, X_half2 = np.array_split(X, 2)
y_half1, y_half2 = np.array_split(y, 2)

# 遍历 weight1 值并生成相应的数据
for weight1 in weight1_values:
    # 对数据进行加权相加
    X_weighted = (X_half1 * weight1 + X_half2 * (1 - weight1))
    y_weighted = (y_half1 * weight1 + y_half2 * (1 - weight1))

    # 将生成的数据添加到列表中
    X_weighted_list.append(X_weighted)
    y_weighted_list.append(y_weighted)

# Concatenate X_weighted_list with the original X data
X_train = np.vstack([X] + X_weighted_list)
# Concatenate y_weighted_list with the original y data
y_train = np.vstack([y] + y_weighted_list)
# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test_original)


# Define the architecture of your neural network
keras.backend.clear_session()
model = Sequential()
model.add(Dense(16, input_dim=5, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='linear'))   #linear activation function


import keras.backend as K
from keras.layers import Dense, Input, Layer
from keras.models import Model

# 定义和编译模型
keras.backend.clear_session()
model = Sequential()
model.add(Dense(16, input_dim=5, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed), bias_initializer=tf.keras.initializers.Zeros()))
model.add(Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed), bias_initializer=tf.keras.initializers.Zeros()))
model.add(Dense(4, activation='linear', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed), bias_initializer=tf.keras.initializers.Zeros()))


# 编译模型
# Compile the model
model.compile(loss='mean_squared_error',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['mae'])

# Define a callback to save the best model
checkpoint_callback = ModelCheckpoint(
    'best_final_test.h5',
    monitor='val_loss',  # Monitor the validation loss
    save_best_only=True,  # Save only the best model
    mode='min',  # In 'min' mode, it will save the model when validation loss is minimized
    verbose=1  # Show progress in training logs
)
# Train the model
model.fit(X_train, y_train,
          epochs=200, batch_size=32, validation_split=0.2,
          verbose=1, shuffle=True,
          callbacks=[checkpoint_callback])


prediction = model.predict(X_test)
mae = mean_absolute_error(prediction, y_test_original)
mse = mean_squared_error(prediction, y_test_original)
rmse = np.sqrt(mse)
print("Mean Absolute Error_l (MAE):", mae)
print("Mean Squared Error_l (MSE):", mse)
print("Root Mean Squared Error_l (RMSE):", rmse)

