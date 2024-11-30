import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder

def clean_df_yann(dataset):
    columns_data = dataset.copy()
    columns = ['piezo_station_department_code', 'piezo_station_bdlisa_codes', 'piezo_measurement_date', 'piezo_producer_name', 'hydro_station_code', 'hydro_hydro_quantity_elab', 'prelev_usage_label_1', 'insee_%_agri']
    
    # Feature 1
    feature = columns_data['piezo_station_department_code'].copy()
    feature = pd.to_numeric(feature, errors='coerce').astype('Int64')
    columns_data['piezo_station_department_code'] = feature

    # Feature 2
    feature = columns_data['piezo_station_bdlisa_codes'].copy()
    feature = feature.apply(lambda x: eval(x) if x.__class__==str else x)
    columns_data['piezo_station_bdlisa_codes'] = feature
    columns_data = columns_data.explode('piezo_station_bdlisa_codes', ignore_index=True)

    # Feature 3
    feature = columns_data['piezo_measurement_date'].copy()
    feature = pd.to_datetime(feature, format=f"%Y-%m-%d")
    columns_data['piezo_measurement_date'] = feature

    # Feature 4
    feature = columns_data['piezo_producer_name']
    columns_data.drop(['piezo_producer_name'], axis=1)

    # Feature 5
    feature = columns_data['insee_%_agri']
    feature = pd.to_numeric(feature, errors='coerce')
    columns_data['insee_%_agri'] = feature

    return columns_data

def add_date_features(df, date_column):
    """
    Given a DataFrame and a datetime column, extract day, month, and year,
    and add them as new columns to the DataFrame.
    
    Args:
    - df: The input DataFrame.
    - date_column: The name of the column containing datetime values.

    Returns:
    - The DataFrame with new columns: <date_column>_day, <date_column>_month, <date_column>_year.
    """
    # Ensure the column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce', format=f"%Y-%m-%d")
    
    # Extract day, month, and year
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_year'] = df[date_column].dt.year
    
    return df

def encoding_yann(dataset):
    # Variables for target encoding
    target_dict = {
    'Very Low':0,
    'Low':1, 
    'Average':2,
    'High':3,
    'Very High':4
    }           
    dataset['num_target'] = dataset['piezo_groundwater_level_category'].map(target_dict)
    target_column = 'num_target'
    target_encoding_vars = ['piezo_station_department_code', 'piezo_station_bdlisa_codes', 'hydro_station_code']
    one_hot_encoding_vars = ['hydro_hydro_quantity_elab', 'prelev_usage_label_1']

    # Perform target encoding
    target_encoder = TargetEncoder(smooth='auto')
    target_encoded_data = target_encoder.fit_transform(dataset[target_encoding_vars], dataset[target_column])
    targ_feat_names = target_encoder.get_feature_names_out()
    temp_df = pd.DataFrame(target_encoded_data)
    temp_df.columns = targ_feat_names
    dataset = pd.concat([dataset, temp_df], axis=1)


    # Perform One-Hot Encoding
    one_hot_encoder = OneHotEncoder()
    oh_encoded_data = one_hot_encoder.fit_transform(dataset[one_hot_encoding_vars]).toarray()
    oh_feat_names = one_hot_encoder.get_feature_names_out()
    temp_df = pd.DataFrame(oh_encoded_data)
    temp_df.columns = oh_feat_names
    dataset = pd.concat([dataset, temp_df], axis=1)

    # Set the encoders for those variables
    encoders_dict = {
        'one_hot_encoder':one_hot_encoder,
        'target_encoder':target_encoder
    }

    variables_dict = {
        'one_hot_encoder_variables':one_hot_encoding_vars,
        'target_encoder_variables':target_encoding_vars
    }

    # Encoding Dates
    dataset = add_date_features(dataset, 'piezo_measurement_date')
    
    return dataset, encoders_dict, variables_dict


def preprocess_and_clean_yann(dataset):
    """
    Takes the dataset as entry, cleans it and return the cleaned version and the encoders objects
    """
    data = clean_df_yann(dataset)
    data, encoders_dict, variables_dict = encoding_yann(data)

    return data, encoders_dict, variables_dict
