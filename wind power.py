# Library Versions:
# os: nt
# logging: 0.5.1.2
# pandas: 2.2.2
# numpy: 1.26.4
# matplotlib: 3.8.2
# keras: 3.5.0
# scikit-learn: 1.5.1
# xgboost: 2.0.3

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Bidirectional, Reshape
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.api.layers import Input
import xgboost as xgb


# Set up logging for better debugging and process tracking
logging.basicConfig(level=logging.INFO)

# Define paths for input data and output results
DATA_PATH = 'vmd.xlsx'  # Path to the input dataset (Excel format)
OUTPUT_PATH = 'F:/D/pycharm/essay/predicted_results.xlsx'  # Path for saving predictions output

# Ensure output directory exists, create it if necessary
output_dir = os.path.dirname(OUTPUT_PATH)
os.makedirs(output_dir, exist_ok=True)


# Function to load dataset from an Excel file
def read_data(file_path, sheet_name='Sheet1'):
    """
    Reads data from an Excel file. Logs success or failure during the loading process.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str): The sheet name to load from the Excel file. Default is 'Sheet1'.

    Returns:
        pandas.DataFrame: Data loaded from the Excel file.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise


# Normalization function to scale data to the range [0, 1]
def normalize(data):
    """
    Normalizes the data to the range [0, 1].

    Args:
        data (numpy.array): Input data to be normalized.

    Returns:
        tuple: Normalized data along with the maximum and minimum values used for rescaling.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data)), np.max(data), np.min(data)


# Function to generate datasets for time-series forecasting
def dataset(data, win_size=96):
    """
    Creates datasets for time series forecasting using a sliding window approach.

    Args:
        data (numpy.array): The normalized input data.
        win_size (int): The window size used to create the time-series datasets.

    Returns:
        tuple: Arrays for input (X) and output (Y) for model training.
    """
    X, Y = [], []
    for i in range(len(data) - win_size):
        X.append(data[i:i + win_size])
        Y.append(data[i + win_size])
    return np.asarray(X), np.asarray(Y)


# Function to build and compile the LSTM model
def build_model(input_shape):
    """
    Builds a model with a single Conv1D layer followed by Bidirectional LSTM and Dense layers for time series forecasting.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        keras.models.Sequential: The compiled model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Convolutional layer
    model.add(Conv1D(filters=64, kernel_size=7, activation='relu'))  # 1D convolution layer
    model.add(MaxPooling1D(pool_size=2))  # Max pooling layer

    # Reshape for Bidirectional LSTM input
    model.add(Reshape((-1, 64)))  # Reshape the data for LSTM input

    # Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(256, activation='relu')))  # Bidirectional LSTM layer

    # Fully connected layers
    model.add(Dense(32, activation='relu'))  # Dense layer with 32 units
    model.add(Dense(16, activation='relu'))  # Dense layer with 16 units
    model.add(Dense(1))  # Output layer with 1 unit (for regression task)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')  # Compile with Adam optimizer and MSE loss

    return model


# Custom XGBoost loss function (including penalty for negative predictions)
def custom_loss(y_true, y_pred):
    """
    A custom loss function that includes a penalty for negative predictions in XGBoost.

    Args:
        y_true (numpy.array): True values of the target variable.
        y_pred (numpy.array): Predicted values from the model.

    Returns:
        tuple: Gradients and Hessians used by XGBoost for training.
    """
    base_loss = (y_true - y_pred) ** 2
    penalty = np.maximum(-y_pred, 0)
    grad = -2 * (y_true - y_pred) + 0.2 * np.where(y_pred < 0, 1, 0)
    hess = np.ones_like(y_pred) * 2
    return grad, hess


# Function to process each IMF component, generate predictions, and store results
def process_imf(df, imf_columns, win_size=96):
    """
    Processes each IMF (Intrinsic Mode Function) component, trains models, and stores results.

    Args:
        df (pandas.DataFrame): The dataframe containing the dataset.
        imf_columns (list): List of IMF column names to process.
        win_size (int): Size of the sliding window for time-series forecasting.

    Returns:
        tuple: DataFrames containing the predicted results for training and testing data.
    """
    predicted_imf_results_train = pd.DataFrame()
    predicted_imf_results_test = pd.DataFrame()
    actual_power = df['Actual_Power'].values  # Target variable for prediction (actual power)

    for imf in imf_columns:
        logging.info(f"Processing {imf}...")

        # Normalize the data
        data, arr_max, arr_min = normalize(np.array(df[imf]))
        data = data.ravel()

        # Generate time-series datasets
        data_x, data_y = dataset(data, win_size=win_size)
        data_x = np.expand_dims(data_x, axis=2)  # Expand dims for the LSTM input format

        # Split data into training and testing sets (80% train, 20% test)
        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False)

        # Build and train the model
        model = build_model((train_x.shape[1], train_x.shape[2]))
        history = model.fit(train_x, train_y, epochs=50, batch_size=1024, validation_split=0.2, shuffle=False,
                            verbose=0)

        # Predict using the trained model
        train_pred = model.predict(train_x)
        test_pred = model.predict(test_x)

        # Rescale the predictions back to the original range
        train_pred_rescaled = train_pred * (arr_max - arr_min) + arr_min
        test_pred_rescaled = test_pred * (arr_max - arr_min) + arr_min

        # Store the predictions
        predicted_imf_results_train[imf] = train_pred_rescaled.ravel()
        predicted_imf_results_test[imf] = test_pred_rescaled.ravel()

    return predicted_imf_results_train, predicted_imf_results_test, actual_power


# Function to save results to an Excel file
def save_results(train_results, test_results, actual_power, output_path):
    """
    Saves the training and testing predictions, along with actual power values, to an Excel file.

    Args:
        train_results (pandas.DataFrame): Predictions for the training set.
        test_results (pandas.DataFrame): Predictions for the testing set.
        actual_power (numpy.array): Actual power values corresponding to the predictions.
        output_path (str): Path to save the output Excel file.
    """
    with pd.ExcelWriter(output_path) as writer:
        # Write training and testing predictions to different sheets in the Excel file
        train_results['Actual_Power'] = actual_power[:len(train_results)]
        test_results['Actual_Power'] = actual_power[len(train_results):len(train_results) + len(test_results)]

        train_results.to_excel(writer, sheet_name='Train_Predictions', index=False)
        test_results.to_excel(writer, sheet_name='Test_Predictions', index=False)


# Function to integrate results with XGBoost and perform GridSearchCV for optimal parameters
def integrate_with_xgboost(train_results, test_results):
    """
    Integrates the model predictions using XGBoost and applies GridSearchCV to find the best hyperparameters.

    Args:
        train_results (pandas.DataFrame): Predictions for the training set.
        test_results (pandas.DataFrame): Predictions for the testing set.

    Returns:
        tuple: Predictions for training and testing sets using the best XGBoost model.
    """
    X_train = train_results.drop(columns=['Actual_Power'])
    y_train = train_results['Actual_Power']
    X_test = test_results.drop(columns=['Actual_Power'])
    y_test = test_results['Actual_Power']

    # Initialize the XGBoost regressor
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 6],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Perform GridSearchCV to find the best model
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Make predictions using the best model
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    logging.info(f"Best parameters found: {best_params}")

    return train_pred, test_pred, y_train, y_test


# Function to plot the results of the predictions
def plot_results(y_train, train_pred, y_test, test_pred):
    """
    Plots the actual vs predicted values for both training and testing datasets.

    Args:
        y_train (numpy.array): Actual values for the training set.
        train_pred (numpy.array): Predicted values for the training set.
        y_test (numpy.array): Actual values for the testing set.
        test_pred (numpy.array): Predicted values for the testing set.
    """
    # Plot for training data
    plt.figure(figsize=(12, 6))
    plt.plot(y_train, label='Actual Train Data')
    plt.plot(train_pred, label='Predicted Train Data')
    plt.title('Train Data Predictions vs Actual')
    plt.legend()
    plt.show()

    # Plot for testing data
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Test Data')
    plt.plot(test_pred, label='Predicted Test Data')
    plt.title('Test Data Predictions vs Actual')
    plt.legend()
    plt.show()


# Main function to execute the entire workflow
def main():
    try:
        # Step 1: Load the dataset
        df = read_data(DATA_PATH)

        # Step 2: Identify IMF components (columns starting with 'imf')
        imf_columns = [col for col in df.columns if col.startswith('imf')]

        # Step 3: Process each IMF component and train models
        predicted_imf_results_train, predicted_imf_results_test, actual_power = process_imf(df, imf_columns)

        # Step 4: Save the results to an Excel file
        save_results(predicted_imf_results_train, predicted_imf_results_test, actual_power, OUTPUT_PATH)

        # Step 5: Integrate predictions using XGBoost and perform GridSearchCV
        train_pred, test_pred, y_train, y_test = integrate_with_xgboost(predicted_imf_results_train,
                                                                        predicted_imf_results_test)

        # Step 6: Plot the results
        plot_results(y_train, train_pred, y_test, test_pred)

        logging.info("Process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


# Execute the main function
if __name__ == "__main__":
    main()
# After a misalignment of the sliding window between the predicted results and the actual results, a manual adjustment of the sliding window is needed to align the data.