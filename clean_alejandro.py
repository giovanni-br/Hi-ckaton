import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, LabelEncoder, StandardScaler

# Define the file path
file_path = './X_train_Hi5.csv'
file_path_test = './X_test_Hi5.csv'

dataset = pd.read_csv(file_path, nrows=100000)
test_dataset = pd.read_csv(file_path_test, nrows=1000)

def encodeAlejandroVariables(dataset):
    # Columns to check
    columns_to_check = [
        'piezo_station_department_name',
        'piezo_station_commune_name',
        'prelev_structure_code_2',
        'piezo_status',
        'piezo_measure_nature_name',
        'hydro_status_label',
        'prelev_usage_label_0',
        'insee_%_ind'
    ]

    # Replace non-numeric values in 'insee_%_ind'
    dataset['insee_%_ind'] = pd.to_numeric(dataset['insee_%_ind'], errors='coerce')
    # Handle missing values introduced during conversion
    dataset['insee_%_ind'].fillna(dataset['insee_%_ind'].mean(), inplace=True)

    # Fill missing values with mode for categorical columns before encoding
    for column in columns_to_check:
        if column in dataset.columns:
            mode_value = dataset[column].mode()[0]  # Get the mode (most frequent value)
            dataset[column].fillna(mode_value, inplace=True)

    # Extract the target column (last column)
    target_column = dataset.columns[-1]

    # Variables for target encoding
    target_encoding_vars = ['piezo_station_department_name', 'piezo_station_commune_name', 'prelev_structure_code_2']

    # Initialize TargetEncoder
    target_encoder = TargetEncoder()

    # Perform target encoding for each variable
    for variable in target_encoding_vars:
        dataset[variable] = target_encoder.fit_transform(
            dataset[variable].values.reshape(-1, 1), dataset[target_column]
        )

    # Variables for one-hot encoding
    one_hot_encoding_vars = [
        'piezo_status', 
        'piezo_measure_nature_name', 
        'hydro_status_label', 
        'prelev_usage_label_0'
    ]

    # Perform one-hot encoding using OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    one_hot_encoded = one_hot_encoder.fit_transform(dataset[one_hot_encoding_vars])

    # Convert the one-hot encoded data into a DataFrame
    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns=one_hot_encoder.get_feature_names_out(one_hot_encoding_vars),
        index=dataset.index
    )

    # Drop the original columns that were one-hot encoded
    dataset = dataset.drop(columns=one_hot_encoding_vars)

    # Concatenate the one-hot encoded columns with the original dataset
    dataset = pd.concat([dataset, one_hot_df], axis=1)

    # Replace missing values in numeric columns with their mean
    for column in ['meteo_radiation_IR', 'meteo_cloudiness', 'meteo_cloudiness_height']:
        if column in dataset.columns:
            dataset[column].fillna(dataset[column].mean(), inplace=True)

    # Select numerical columns
    numericalDataset = dataset.select_dtypes(include=[np.number])

    # Create dictionaries for encoders and variables
    encoders_dict = {
        'one_hot_encoder': one_hot_encoder,
        'target_encoder': target_encoder
    }

    variables_dict = {
        'one_hot_encoder_variables': one_hot_encoding_vars,
        'target_encoder_variables': target_encoding_vars
    }

    return numericalDataset, encoders_dict, variables_dict