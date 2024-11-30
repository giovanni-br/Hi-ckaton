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

    # Replace non-numeric values
    dataset['insee_%_ind'] = pd.to_numeric(dataset['insee_%_ind'], errors='coerce')
    # Handle missing values introduced during conversion
    dataset['insee_%_ind'].fillna(dataset['insee_%_ind'].mean(), inplace=True)

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

    # Perform one-hot encoding using OneHotEncoder only on existing columns
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    one_hot_encoded = one_hot_encoder.fit_transform(dataset[one_hot_encoding_vars])

    # Replace the original columns with one-hot encoded columns
    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns=one_hot_encoder.get_feature_names_out(one_hot_encoding_vars),
        index=dataset.index
    )

    # Now replace the original columns with the one-hot encoded columns
    for i, column in enumerate(one_hot_encoding_vars):
        dataset[column] = one_hot_df.iloc[:, i]

    # Replace missing values with the average for specific columns
    for col in columns_to_check:
        dataset[col].fillna(dataset[col].mean(), inplace=True)

    # Replace missing values in specific numeric columns with their mean
    for column in ['meteo_radiation_IR', 'meteo_cloudiness', 'meteo_cloudiness_height']:
        if column in dataset.columns:
            dataset[column].fillna(dataset[column].mean(), inplace=True)

    # Create dictionaries for encoders and variables
    encoders_dict = {
        'one_hot_encoder': one_hot_encoder,
        'target_encoder': target_encoder
    }

    variables_dict = {
        'one_hot_encoder_variables': one_hot_encoding_vars,
        'target_encoder_variables': target_encoding_vars
    }

    return dataset, encoders_dict, variables_dict