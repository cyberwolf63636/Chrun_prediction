import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import io

def load_data(uploaded_file):
    """
    Load data from an uploaded CSV file.
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        The uploaded CSV file
    
    Returns:
    --------
    pandas.DataFrame
        The loaded dataset
    """
    try:
        # Convert streamlit UploadedFile to bytes
        bytes_data = uploaded_file.getvalue()
        
        # Convert to BytesIO object
        bytes_io = io.BytesIO(bytes_data)
        
        # Try to read with different encodings and separators
        try:
            df = pd.read_csv(bytes_io)
        except:
            # Reset stream position
            bytes_io.seek(0)
            try:
                df = pd.read_csv(bytes_io, encoding='latin1')
            except:
                # Reset stream position
                bytes_io.seek(0)
                df = pd.read_csv(bytes_io, sep=';')
        
        return df
    except Exception as e:
        raise Exception(f"Failed to load data: {str(e)}")

def preprocess_data(data):
    """
    Preprocess the dataset for machine learning.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The raw dataset
    
    Returns:
    --------
    pandas.DataFrame, list
        The preprocessed dataset and list of feature columns
    """
    try:
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Check if 'Churn' column exists and convert to binary if needed
        if 'Churn' in df.columns:
            if df['Churn'].dtype == object:
                # If Churn is categorical (Yes/No, True/False, etc.)
                churn_mapping = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0, True: 1, False: 0}
                df['Churn'] = df['Churn'].map(lambda x: churn_mapping.get(x, x))
                # If still object, use label encoder
                if df['Churn'].dtype == object:
                    le = LabelEncoder()
                    df['Churn'] = le.fit_transform(df['Churn'])
            # Ensure Churn is numeric
            df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
            df['Churn'] = df['Churn'].fillna(0)
            df['Churn'] = df['Churn'].astype(int)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Fill missing values
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # Encode categorical variables
        for col in categorical_cols:
            if col != 'Churn':  # Skip if it's the target column
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        
        # Get list of feature columns (all columns except Churn)
        feature_columns = [col for col in df.columns if col != 'Churn']
        
        return df, feature_columns
    
    except Exception as e:
        st.error(f"Error in data preprocessing: {str(e)}")
        raise Exception(f"Failed to preprocess data: {str(e)}")
