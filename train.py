import lightgbm
import pandas as pd
import pickle

# Specifying paths

# Path to input data sets
input_path = '/opt/ml/input/data'
# Path to folder with model
model_path = '/opt/ml/model'
# Path to file with model hyperparameters
param_path = '/opt/ml/input/config/hyperparameters.json'

# Selecting channel type
train_channel_name = 'train'
test_channel_name = 'test'
# Selecting path to datasets
train_path = input_path + '/' + train_channel_name
test_path = input_path + '/' + test_channel_name

# Reading data
data_train = pd.read_csv(train_path + '/' + 'train.csv')
data_test = pd.read_csv(test_path + '/' + 'test.csv')
X = data_train.iloc[:, [0, 1, 2, 3]]
y = data_train.iloc[:, 4].values
X_test = data_test.iloc[:, [0, 1, 2, 3]]
y_test = data_test.iloc[:, 4].values

# Training model
model = lightgbm.LGBMRegressor()
model.fit(X, y)
predictions = model.predict(X_test)


# Saving data to folders from which this data will be uploaded to S3 bucket
with open(model_path + '/trained_model.pickle', 'wb') as f:
    pickle.dump(model, f)

with open(model_path + '/' + 'predictions.pickle', 'wb') as f:
    pickle.dump(predictions, f)