import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, TargetEncoder

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

    # Convert categorical labels in the target column into numerical values
    label_encoder_target = LabelEncoder()
    dataset[target_column] = label_encoder_target.fit_transform(dataset[target_column])

    # Variables for target encoding
    target_encoding_vars = ['piezo_station_department_name', 'piezo_station_commune_name', 'prelev_structure_code_2']

    # Initialize TargetEncoder
    target_encoder = TargetEncoder()

    # Perform target encoding for each variable
    for variable in target_encoding_vars:
        # Reshape the variable to 2D for TargetEncoder
        dataset[variable] = target_encoder.fit_transform(dataset[variable].values.reshape(-1, 1), dataset[target_column])

    # Variables for label encoding
    label_encoding_vars = [
        'piezo_status', 
        'piezo_measure_nature_name', 
        'hydro_status_label', 
        'prelev_usage_label_0'
    ]

    label_encoders = {}
    for variable in label_encoding_vars:
        encoder = LabelEncoder()
        dataset[variable] = encoder.fit_transform(dataset[variable])
        label_encoders[variable] = encoder  # Store encoders for inverse transformation if needed

    # Keep only numerical columns in numericalDataset, including those transformed and others
    numericalDataset = dataset.select_dtypes(include=[np.number])
    #numericalDataset = dataset

    # Fill null, NaN, or None values with the average value of the column
    numericalDataset.fillna(numericalDataset.mean(), inplace=True)

    # Fill the column with a default value (e.g., 0)
    numericalDataset['meteo_radiation_IR'] = numericalDataset['meteo_radiation_IR'].fillna(0)

    # Fill the column with a default value (e.g., 0)
    numericalDataset['meteo_cloudiness'] = numericalDataset['meteo_cloudiness'].fillna(0)

    # Fill the column with a default value (e.g., 0)
    numericalDataset['meteo_cloudiness_height'] = numericalDataset['meteo_cloudiness_height'].fillna(0)

    return numericalDataset, label_encoders, target_encoder, label_encoder_target