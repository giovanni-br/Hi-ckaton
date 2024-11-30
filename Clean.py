import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder

# Load data
file_path = '/home/jovyan/my_work/X_train_Hi5.csv'
data = pd.read_csv(file_path, nrows=100000)

# List of categorical columns
categorical_gio = [
    'piezo_station_bss_code', 'piezo_obtention_mode', 
    'piezo_measure_nature_code', 'hydro_observation_date_elab', 
    'prelev_structure_code_0', 'prelev_volume_obtention_mode_label_1', 
    'insee_med_living_level'
]

# Clean BSS Code
def clean_bss_code_old(code):
    return code.split("/")[1] if isinstance(code, str) and "/" in code else code

data['piezo_station_bss_code'] = data['piezo_station_bss_code'].apply(clean_bss_code_old)

# Handle dates
data['hydro_observation_date_elab'] = pd.to_datetime(data['hydro_observation_date_elab'], errors='coerce')

def add_date_features(df, date_column):
    # Ensure the column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    # Extract day, month, and year
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_year'] = df[date_column].dt.year
    return df

data = add_date_features(data, 'hydro_observation_date_elab')

# Convert and fill missing values for 'insee_med_living_level'
def convert_to_int_with_nan(value):
    try:
        return int(value)  # Try converting to integer
    except ValueError:
        return np.nan  # Return NaN if conversion fails

data['insee_med_living_level'] = data['insee_med_living_level'].apply(convert_to_int_with_nan)
median_value = data['insee_med_living_level'].median()
data['insee_med_living_level'] = data['insee_med_living_level'].fillna(median_value)

# Encoding strategy
# Define columns for target and one-hot encoding
target_encode_columns = ['piezo_station_bss_code', 'prelev_structure_code_0']
one_hot_encode_columns = [
    col for col in categorical_gio 
    if col not in target_encode_columns 
    and col != 'hydro_observation_date_elab' 
    and col != 'insee_med_living_level'
]

# Target column (categorical target)
target_column = 'piezo_groundwater_level_category'  # Replace with the actual target column name
if target_column not in data.columns:
    raise ValueError(f"Target column '{target_column}' not found in the dataset.")

# Encode categorical target
label_encoder = LabelEncoder()
data[target_column] = label_encoder.fit_transform(data[target_column])

# Use TargetEncoder from sklearn
target_encoder = TargetEncoder()

# Perform target encoding for each variable
for variable in target_encode_columns:
    # Reshape the variable to 2D for TargetEncoder
    data[variable] = target_encoder.fit_transform(
        data[variable].values.reshape(-1, 1), 
        data[target_column]
    )

# Perform one-hot encoding for selected columns
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
one_hot_encoded = pd.DataFrame(
    one_hot_encoder.fit_transform(data[one_hot_encode_columns]),
    columns=one_hot_encoder.get_feature_names_out(one_hot_encode_columns),
    index=data.index
)

# Drop original one-hot encoded columns and append the new ones
data = data.drop(columns=one_hot_encode_columns).join(one_hot_encoded)

# Final DataFrame
print(data.head())
