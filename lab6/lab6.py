import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

file_path_prices = "/Users/luiza/Desktop/IA/Lab6/data/prices.npy"
file_path_training_data = "/Users/luiza/Desktop/IA/Lab6/data/training_data.npy"


training_data = np.load(file_path_training_data)
prices = np.load(file_path_prices)
training_data = training_data[0:]
prices = prices[0:]

def normalize(train, test):
    scaler = preprocessing.StandardScaler()
    scaler.fit(train)
    standard_train = scaler.transform(train)
    standard_test = scaler.transform(test)
    return standard_train, standard_test

num_samples_fold = len(training_data) // 3
training1, prices1 = training_data[:num_samples_fold], prices[:num_samples_fold]
training2, prices2 = training_data[num_samples_fold: 2 * num_samples_fold],\
                     prices[num_samples_fold: 2 * num_samples_fold]
training3, prices3 = training_data[2 * num_samples_fold:], prices[2 * num_samples_fold:]

train_data_1, test1 = normalize(training1, training2)
train_data_2, test2 = normalize(training2, training3)
train_data_3, test3 = normalize(training1, training3)

linear_regression_model = LinearRegression()
linear_regression_model.fit(train_data_1, prices1)
pred_1_linear = linear_regression_model.predict(test1)
mse_value1_linear = mean_squared_error(prices1, pred_1_linear)
mae_value1_linear = mean_absolute_error(prices1, pred_1_linear)

linear_regression_model.fit(train_data_2, prices2)
pred_2_linear = linear_regression_model.predict(test2)
mse_value2_linear = mean_squared_error(prices2, pred_2_linear)
mae_value2_linear = mean_absolute_error(prices2, pred_2_linear)

linear_regression_model.fit(train_data_3, prices3)
pred_3_linear = linear_regression_model.predict(test3)
mse_value3_linear = mean_squared_error(prices3, pred_3_linear)
mae_value3_linear = mean_absolute_error(prices3, pred_3_linear)

mean_mse_value_linear = (mse_value1_linear + mse_value2_linear + mse_value3_linear)/3
mean_mae_value_linear = (mae_value1_linear + mae_value2_linear + mae_value3_linear)/3

print("2) MAE and MSE: ")
print(mean_mae_value_linear, mean_mse_value_linear)

ridge_regression_model = Ridge()

ridge_regression_model.fit(train_data_1, prices1)
pred1 = ridge_regression_model.predict(test1)
mse_value1 = mean_squared_error(prices1, pred1)
mae_value1 = mean_absolute_error(prices1, pred1)


ridge_regression_model.fit(train_data_2, prices2)
pred2 = ridge_regression_model.predict(test2)
mse_value2 = mean_squared_error(prices2, pred2)
mae_value2 = mean_absolute_error(prices2, pred2)

ridge_regression_model.fit(train_data_3, prices3)
pred3 = ridge_regression_model.predict(test3)
mse_value3 = mean_squared_error(prices3, pred3)
mae_value3 = mean_absolute_error(prices3, pred3)

mean_mse_value_ridge = (mse_value1 + mse_value2 + mse_value3)/3
mean_mae_value_ridge = (mae_value1 + mae_value2 + mae_value3)/3

print("3) MAE and MSE: ")
print(mean_mae_value_ridge, mean_mse_value_ridge)

