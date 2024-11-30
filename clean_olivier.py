import pandas as pd
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

# Fonction I'm using written by Giovanni:
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
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    # Extract day, month, and year
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_year'] = df[date_column].dt.year
    return df


def clean_olivier(data):
    """
    Clean and encode the following variables:
    - 'piezo_station_pe_label': dropped
    - 'piezo_bss_code': simplified down to 760 categories, target encoded
    - 'piezo_qualification': kept only the rows with 'Correcte' before deleting the variable
    - 'meteo_date': split onto meteo_date_{day, month, year}
    - 'hydro_qualification_label': kept as is and label encoded
    - 'prelev_structure_code_1': kept and target encoded
    - 'prelev_volume_obtention_mode_label_2': kept and label encoded
    Returning: dataset, encoders_dict, variables_dict
    """

    vars = ['piezo_station_pe_label', 'piezo_bss_code', 'piezo_continuity_name', 'meteo_date',
            'hydro_qualification_label', 'prelev_structure_code_1', 'prelev_volume_obtention_mode_label_2']

    # Fonctions I created for the cleaning:
    def clean_time_variable_longtype(variable_name):
        """Cleans time variables with a format like "Sun Jul 14 13:00:02 CEST 2024", turning it to datetime."""

        def CEST(string):
            return string.replace("CEST", "+0200")

        def CET(string):
            return string.replace("CET", "+0100")

        data[variable_name] = data[variable_name].apply(CET)
        data[variable_name] = data[variable_name].apply(CEST)

        data[variable_name] = pd.to_datetime(data[variable_name], format="%a %b %d %H:%M:%S %z %Y")

    def clean_bss_code_old(code):
        """Remove coordinates part of bss code"""
        return code.split("/")[1]

    def clean_time_variable_shorttype(variable_name):
        """Cleans time variable with a format like "2020-01-24"""
        data[variable_name] = pd.to_datetime(data[variable_name], format="%Y-%m-%d")

    # CLEANING:
    data["piezo_bss_code"] = data["piezo_bss_code"].apply(clean_bss_code_old)
    clean_time_variable_longtype("piezo_station_update_date")

    # Cleaning piezo_qualifications, keeping only Correcte ones.
    data = data[data["piezo_qualification"] == "Correcte"]
    del data["piezo_qualification"]
    # Removing label of piezo stations -> we have the code, better to get it
    del data["piezo_station_pe_label"]

    clean_time_variable_shorttype("meteo_date")

    # ENCODING

    data = add_date_features(data, "meteo_date")
    del data["meteo_date"]
    data = add_date_features(data, "piezo_station_update_date")
    del data["piezo_station_update_date"]

    # Encoding the target
    target_dict = {
        'Very Low':0,
        'Low':1, 
        'Average':2,
        'High':3,
        'Very High':4
        }           
    data['num_target'] = data['piezo_groundwater_level_category'].map(target_dict)

    # Extract the target column (last column)
    target_column = "num_target"

    # Variables for target encoding
    target_encoding_vars = ['piezo_bss_code', "prelev_structure_code_1"]

    # Perform target encoding
    target_encoder = TargetEncoder(smooth='auto')
    target_encoded_data = target_encoder.fit_transform(data[target_encoding_vars], data[target_column])
    targ_feat_names = target_encoder.get_feature_names_out()
    temp_df = pd.DataFrame(target_encoded_data)
    temp_df.columns = targ_feat_names
    data = pd.concat([data, temp_df], axis=1)

    # Variables for one hot encoding
    one_hot_encoding_vars = [
        'hydro_qualification_label',
        "prelev_volume_obtention_mode_label_2"
    ]

    # Perform One-Hot Encoding
    one_hot_encoder = OneHotEncoder()
    oh_encoded_data = one_hot_encoder.fit_transform(data[one_hot_encoding_vars]).toarray()
    oh_feat_names = one_hot_encoder.get_feature_names_out()
    temp_df = pd.DataFrame(oh_encoded_data)
    temp_df.columns = oh_feat_names
    data = pd.concat([data, temp_df], axis=1)

    encoders_dict = {
        'one_hot_encoder': one_hot_encoder,
        'target_encoder': target_encoder
    }

    variables_dict = {
        'one_hot_encoder_variables': one_hot_encoding_vars,
        'target_encoder_variables': target_encoding_vars
    }

    return data, encoders_dict, variables_dict

