import pandas as pd
import numpy as np

def _create_price_rank_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks hotels by price_usd within each search (srch_id).
    Lower price gets a higher rank (e.g., rank 1 is the cheapest).
    Handles ties by assigning the average rank.
    A new column 'price_rank_within_search' is added to the DataFrame.
    """
    df['price_rank_within_search'] = df.groupby('srch_id')['price_usd'].rank(method='average', ascending=True)
    return df

def _create_value_for_money_score(df: pd.DataFrame, max_score: float = 100.0) -> pd.DataFrame:
    """
    Creates a value-for-money score: prop_review_score / price_usd.
    Handles cases where price_usd is 0 or NaN.
    New column 'value_for_money_score' is added.
    """
    df['value_for_money_score'] = 0.0  # Initialize column

    # Case 1: price_usd > 0
    price_positive_mask = df['price_usd'] > 0
    # Ensure prop_review_score is also not NaN for these calculations if it can be
    # (already handled by preprocessing which fills prop_review_score NaNs with 0)
    df.loc[price_positive_mask, 'value_for_money_score'] = \
        df.loc[price_positive_mask, 'prop_review_score'] / df.loc[price_positive_mask, 'price_usd']

    # Case 2: price_usd == 0 and prop_review_score > 0 (free hotel with good reviews)
    # Assign a fixed high score as a placeholder for exceptional value.
    free_good_review_mask = (df['price_usd'] == 0) & (df['prop_review_score'] > 0)
    df.loc[free_good_review_mask, 'value_for_money_score'] = max_score

    # Fill any NaNs that might have occurred (e.g., if price_usd was NaN or for 0/0 cases if not covered)
    # price_positive_mask handles division by zero for price_usd. 
    # If price_usd was NaN, it wouldn't satisfy price_positive_mask or free_good_review_mask,
    # so value_for_money_score would remain 0.0 (from initialization) or become NaN then 0.0.
    df['value_for_money_score'] = df['value_for_money_score'].fillna(0.0)
    
    return df

def _create_domestic_travel_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a binary feature 'is_domestic_travel'.
    1 if visitor_location_country_id == prop_country_id, 0 otherwise.
    Handles NaN in ID columns by defaulting to 0 (not domestic).
    """
    # Direct comparison produces True, False, or NaN if any operand is NaN.
    comparison_result = (df['visitor_location_country_id'] == df['prop_country_id'])
    
    # Convert boolean to float (True->1.0, False->0.0) to handle NaNs, then fill NaNs with 0.
    df['is_domestic_travel'] = comparison_result.astype(float).fillna(0).astype(int)
    return df

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps to the DataFrame.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame with engineered features.
    """
    df_engineered = df.copy() # Work on a copy to avoid modifying the original DataFrame
    
    # 1. Create price rank feature
    df_engineered = _create_price_rank_feature(df_engineered)
    
    # 2. Create value-for-money score
    df_engineered = _create_value_for_money_score(df_engineered)

    # 3. Create domestic travel feature
    df_engineered = _create_domestic_travel_feature(df_engineered)

    return df_engineered