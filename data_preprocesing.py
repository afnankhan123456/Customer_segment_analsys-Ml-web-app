import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    """
    Ye function Flask se dataframe lega,
    clean karega aur encoded dataframe return karega.
    """
    # Step 1: Fill missing values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Step 2: Remove duplicates and convert dtypes
    df.drop_duplicates(inplace=True)
    df = df.convert_dtypes()

    # Step 3: Drop customer_id if exists
    df_encoded = df.copy()
    if 'customer_id' in df_encoded.columns:
        df_encoded = df_encoded.drop('customer_id', axis=1)

    # Step 4: Encode columns
    for col in df_encoded.select_dtypes(include=['object', 'string']).columns:
        unique_vals = df_encoded[col].nunique()
        if unique_vals <= 5:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        else:
            freq_map = df_encoded[col].value_counts().to_dict()
            df_encoded[col] = df_encoded[col].map(freq_map)

    return df_encoded
