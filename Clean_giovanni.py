import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder

data = pd.read_csv("X_train_Hi5.csv")

def preprocess_dataset(
    dataset,
    date_columns_longtype=None,
    date_columns_shorttype=None,
    categorical_columns=None,
    target_column=None,
    target_encoding_vars=None,
    label_encoding_vars=None,
    one_hot_encode_columns=None
):
    # Data Cleaning Functions
    def clean_time_variable_longtype(variable_name, df):
        """Cleans time variables with a long format, turning them to datetime."""
        def CEST(string):
            return string.replace("CEST", "+0200")
        def CET(string):
            return string.replace("CET", "+0100")
        df[variable_name] = df[variable_name].apply(CET)
        df[variable_name] = df[variable_name].apply(CEST)
        df[variable_name] = pd.to_datetime(df[variable_name], format="%a %b %d %H:%M:%S %z %Y")

    def clean_time_variable_shorttype(variable_name, df):
        """Cleans time variables with a short format, turning them to datetime."""
        df[variable_name] = pd.to_datetime(df[variable_name], format="%Y-%m-%d")

    def add_date_features(df, date_column):
        """Adds day, month, and year features from a datetime column."""
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df[f'{date_column}_day'] = df[date_column].dt.day
        df[f'{date_column}_month'] = df[date_column].dt.month
        df[f'{date_column}_year'] = df[date_column].dt.year
        return df

    # Date Cleaning
    if date_columns_longtype:
        for col in date_columns_longtype:
            clean_time_variable_longtype(col, dataset)
            dataset = add_date_features(dataset, col)
            del dataset[col]
    
    if date_columns_shorttype:
        for col in date_columns_shorttype:
            clean_time_variable_shorttype(col, dataset)
            dataset = add_date_features(dataset, col)
            del dataset[col]
    
    # Cleaning Categorical Columns
    if categorical_columns:
        for col in categorical_columns:
            dataset[col] = dataset[col].apply(lambda x: x.split("/")[1] if isinstance(x, str) and "/" in x else x)
    
    # Target Encoding
    if target_column and target_encoding_vars:
        target_encoder = TargetEncoder()
        for variable in target_encoding_vars:
            dataset[variable] = target_encoder.fit_transform(dataset[[variable]], dataset[target_column])
    
    # Label Encoding
    if label_encoding_vars:
        label_encoders = {}
        for variable in label_encoding_vars:
            encoder = LabelEncoder()
            dataset[variable] = encoder.fit_transform(dataset[variable])
            label_encoders[variable] = encoder  # Store encoder if inverse transform needed
    
    # One-Hot Encoding
    if one_hot_encode_columns:
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        one_hot_encoded = pd.DataFrame(
            one_hot_encoder.fit_transform(dataset[one_hot_encode_columns]),
            columns=one_hot_encoder.get_feature_names_out(one_hot_encode_columns),
            index=dataset.index
        )
        dataset = dataset.drop(columns=one_hot_encode_columns).join(one_hot_encoded)

    return dataset
