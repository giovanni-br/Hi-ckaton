import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def clean_df_Yassine(df : pd.DataFrame) :
    """
    clean the following variables : insee_%_const
                                    prelev_volume_obtention_mode_label_0 
                                    prelev_usage_label_2
                                    piezo_continuity_code
                                    meteo_name
    """

    #clean feature insee_%_const
    df['insee_%_const'] = pd.to_numeric(df['insee_%_const'], errors='coerce')
    df['insee_%_const'] = df['insee_%_const'].fillna(df['insee_%_const'].mean())


    #clean feature piezo_continuity_code
    df['piezo_continuity_code'] = df['piezo_continuity_code'].fillna(df['piezo_continuity_code'].mode()[0])

    #remove feature meteoname
    df = df.drop('meteo_name', axis = 1)

    #remove feature hydro_method_label
    df = df.drop('hydro_method_label', axis = 1)

    #clean prelev_volume_obtention_mode_label_0
    df['prelev_volume_obtention_mode_label_0'] = df['prelev_volume_obtention_mode_label_0'].fillna(df['prelev_volume_obtention_mode_label_0'].mode()[0])

    #clean prelev_usage_label_2
    df['prelev_usage_label_2'] = df['prelev_usage_label_2'].fillna(df['prelev_usage_label_2'].mode()[0])
    
    return df
    

def encoding_Yassine(df) :
#variables for one_hot encoding
    one_hot_encoding_features = [
        "prelev_volume_obtention_mode_label_0",
        'prelev_usage_label_2'
    ]

#perform one-hot encoding
    one_hot_encoder = OneHotEncoder()
    oh_encoded_data = one_hot_encoder.fit_transform(df[one_hot_encoding_features]).toarray()
    oh_feat_names = one_hot_encoder.get_feature_names_out()
    temp_df = pd.DataFrame(oh_encoded_data)
    temp_df.columns = oh_feat_names
    dataset = pd.concat([df, temp_df], axis=1)
        
#dictionnary of encoders and features
    encoders_dict = {'one_hot_encoder' : one_hot_encoder}
    encoded_features = { 'one_hot_encoded_features' : one_hot_encoding_features, 'target_encoded_features' : 0}

    return dataset,encoders_dict, encoded_features


def preprocess_and_clean_Yassine(dataset):
    """
    Takes the dataset as entry, cleans it and return the cleaned version and the encoders objects
    """
    data = clean_df_Yassine(dataset)
    data, encoders_dict, variables_dict = encoding_Yassine(data)

    return data, encoders_dict, variables_dict



    
