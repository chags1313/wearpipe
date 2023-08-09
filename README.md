# Wearpipe Documentation
### Overview
This library offers tools for preparing and analyzing time-series data. It provides utility functions for preprocessing data, training machine learning models with specific architectures (such as LSTM), and assessing their performance. This library is particularly tailored towards sequence data and makes extensive use of popular libraries such as TensorFlow and Scikit-Learn.

### Import Dependencies
python
'from wearpipe import *'

### Functions

#### train_model

Parameters:

features (numpy array): The input data for the model.
labels (numpy array): The target data for the model.
n_classes (int): The number of unique classes.
sequence_length (int, default=100): The length of the input sequences.
... [additional parameters] ...
random_state (int, default=42): Seed for reproducibility.
Returns:

model (Keras Sequential Model): The trained LSTM model.

#### apply_model

Parameters:

df (DataFrame): The original data.
feature_columns (list): Names of the feature columns.
model (Keras model): The trained LSTM model.
le (LabelEncoder): The label encoder used during training.
Returns:

df (DataFrame): Augmented with additional columns for predictions and their confidence.

#### choi_non_wear
Identifies periods of wear and non-wear in accelerometer data.

Parameters:

df (DataFrame): The original data.
accx, accy, accz (str): Column names for the accelerometer data.
sampling_rate (int): Number of data points per minute.
... [additional parameters] ...
Returns:

df (DataFrame): Augmented with a 'Wear' column indicating periods of wear and non-wear.


#### preprocess_data
Preprocesses training data by encoding labels and extracting features.

Parameters:

train_data (DataFrame): The data to preprocess.
class_column (str): Name of the column with class labels.
feature_columns (list): Names of the feature columns.
Returns:

features (numpy array): Extracted features.
labels_encoded (numpy array): Encoded labels.
n_classes (int): Number of unique classes.
le (LabelEncoder): Used for encoding the labels.


#### assess_performance
Assesses the performance of model predictions in a DataFrame.

Parameters:

df (DataFrame): The original data.
true_labels_col (str): Column name with true labels.
predicted_labels_col (str): Column name with predicted labels.
Returns:

metrics (Dict): Contains performance metrics.
conf_mat (Numpy array): The confusion matrix.


#### train_test_split_time
Splits data into training and testing sets based on time.

Parameters:

df (DataFrame): The original data.
min_length (int, default=100): Minimum length of data sequence.
train_ratio (float, default=0.8): Ratio of training data to total data.
Returns:

train (DataFrame): Training data.
test (DataFrame): Testing data.
End of Documentation
This library is designed to aid users in handling, analyzing, and making predictions on sequence data using various machine learning tools. Proper understanding of each function's parameters and return values is crucial for effective utilization.
