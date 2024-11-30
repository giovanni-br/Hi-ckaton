import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
    """

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

    # Extract the target column (last column)
    target_column = data.columns[-1]

    # Variables for target encoding
    target_encoding_vars = ['piezo_bss_code', "prelev_structure_code_1"]

    # Perform target encoding
    for variable in target_encoding_vars:
        target_means = data.groupby(variable)[target_column].mean()
        data[variable] = data[variable].map(target_means)
        data[variable].fillna(data[target_column].mean(), inplace=True)  # Handle missing values

    # Variables for label encoding
    label_encoding_vars = [
        'hydro_qualification_label',
        "prelev_volume_obtention_mode_label_2"
    ]

    label_encoders = {}
    for variable in label_encoding_vars:
        encoder = LabelEncoder()
        data[variable] = encoder.fit_transform(data[variable])
        label_encoders[variable] = encoder  # Store encoders for inverse transformation if needed

    return data
