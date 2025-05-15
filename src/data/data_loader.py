import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def load_data(data_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the dataset from the given path.
    
    Args:
        data_path (str): Path to the data file.
        nrows (Optional[int]): If provided, load only the first 'nrows' of the data.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(data_path, nrows=nrows)
        print(f"Successfully loaded data from {data_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # Consider raising the error or returning None based on desired handling upstream
        raise # Reraise the exception to be handled by the caller
    except Exception as e:
        print(f"An error occurred while loading data from {data_path}: {e}")
        raise # Reraise

def get_feature_types(df: pd.DataFrame) -> dict:
    """
    Get the data types of features in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing feature types
    """
    feature_types = {
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'numerical': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'boolean': df.select_dtypes(include=['bool']).columns.tolist(),
        'datetime': df.select_dtypes(include=['datetime']).columns.tolist()
    }
    return feature_types

def get_missing_value_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missing value statistics for the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame containing missing value statistics
    """
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_stats = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    
    return missing_stats[missing_stats['Missing Values'] > 0].sort_values('Percentage', ascending=False)

def get_basic_stats(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics for the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing basic statistics
    """
    stats = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # in MB
        'duplicates': df.duplicated().sum(),
        'unique_values': {col: df[col].nunique() for col in df.columns}
    }
    return stats

def split_data_by_search_id(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets based on search_id to maintain search groups.
    
    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of search_ids to include in test set
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes
    """
    unique_search_ids = df['srch_id'].unique()
    test_search_ids = np.random.choice(unique_search_ids, 
                                     size=int(len(unique_search_ids) * test_size),
                                     replace=False)
    
    train_mask = ~df['srch_id'].isin(test_search_ids)
    test_mask = df['srch_id'].isin(test_search_ids)
    
    return df[train_mask], df[test_mask]
