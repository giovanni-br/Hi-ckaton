import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder


def clean_data_gio(dataset):
    """
    Cleans the dataset by processing specific features and applying appropriate transformations.
    """
    columns_data = dataset.copy()
    
    # Feature 1: Clean and transform 'piezo_station_bss_code'
    def clean_bss_code_old(code):
        return code.split("/")[1] if isinstance(code, str) and "/" in code else code
    
    columns_data['piezo_station_bss_code'] = columns_data['piezo_station_bss_code'].apply(clean_bss_code_old)
    
    # Feature 2: Handle 'hydro_observation_date_elab'
    def add_date_features(df, date_column):
        """
        Extracts day, month, and year from a datetime column.
        """
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df[f'{date_column}_day'] = df[date_column].dt.day
        df[f'{date_column}_month'] = df[date_column].dt.month
        df[f'{date_column}_year'] = df[date_column].dt.year
        return df
    
    columns_data = add_date_features(columns_data, 'hydro_observation_date_elab')
    
    # Feature 3: Clean 'insee_med_living_level'
    def convert_to_int_with_nan(value):
        try:
            return int(value)  # Convert to integer
        except ValueError:
            return np.nan  # Return NaN if conversion fails
    
    columns_data['insee_med_living_level'] = columns_data['insee_med_living_level'].apply(convert_to_int_with_nan)
    median_value = columns_data['insee_med_living_level'].median()
    columns_data['insee_med_living_level'] = columns_data['insee_med_living_level'].fillna(median_value)
    
    return columns_data


def encode_data_gio(dataset):
    """
    Applies target encoding, label encoding, and one-hot encoding to the dataset.
    """
    # Define the target column
    target_column = 'piezo_groundwater_level_category'
    
    # Check for target column
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    # Encode target column using LabelEncoder
    label_encoder = LabelEncoder()
    dataset[target_column] = label_encoder.fit_transform(dataset[target_column])
    
    # Variables for target encoding
    target_encoding_vars = ['piezo_station_bss_code', 'prelev_structure_code_0']
    target_encoder = TargetEncoder(smooth='auto')
    
    # Perform target encoding
    target_encoded_data = target_encoder.fit_transform(
        dataset[target_encoding_vars],
        dataset[target_column]
    )
    target_encoded_columns = target_encoder.get_feature_names_out()
    target_encoded_df = pd.DataFrame(target_encoded_data, columns=target_encoded_columns, index=dataset.index)
    dataset = pd.concat([dataset, target_encoded_df], axis=1)

    # Variables for one-hot encoding
    one_hot_encode_columns = [
        'piezo_obtention_mode', 'piezo_measure_nature_code',
        'prelev_volume_obtention_mode_label_1'
    ]
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Perform one-hot encoding
    one_hot_encoded = pd.DataFrame(
        one_hot_encoder.fit_transform(dataset[one_hot_encode_columns]),
        columns=one_hot_encoder.get_feature_names_out(one_hot_encode_columns),
        index=dataset.index
    )
    
    # Drop original columns and append one-hot encoded data
    dataset = dataset.drop(columns=one_hot_encode_columns).join(one_hot_encoded)
    
    # Prepare dictionaries for encoders and variables
    encoders_dict = {
        'label_encoder': label_encoder,
        'target_encoder': target_encoder,
        'one_hot_encoder': one_hot_encoder
    }
    
    variables_dict = {
        'target_encoding_vars': target_encoding_vars,
        'one_hot_encode_columns': one_hot_encode_columns
    }
    
    return dataset, encoders_dict, variables_dict


def preprocess_and_clean_gio(dataset):
    """
    Orchestrates the cleaning and encoding of the dataset.
    Returns the preprocessed dataset along with encoder and variable dictionaries.
    """
    data = clean_data_yann(dataset)
    data, encoders_dict, variables_dict = encode_data_yann(data)
    return data, encoders_dict, variables_dict