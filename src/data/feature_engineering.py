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
    df['value_for_money_score'] = 0.0
    price_positive_mask = df['price_usd'] > 0
    df.loc[price_positive_mask, 'value_for_money_score'] = \
        df.loc[price_positive_mask, 'prop_review_score'] / df.loc[price_positive_mask, 'price_usd']
    free_good_review_mask = (df['price_usd'] == 0) & (df['prop_review_score'] > 0)
    df.loc[free_good_review_mask, 'value_for_money_score'] = max_score
    df['value_for_money_score'] = df['value_for_money_score'].fillna(0.0)
    return df

def _create_domestic_travel_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a binary feature 'is_domestic_travel'.
    1 if visitor_location_country_id == prop_country_id, 0 otherwise.
    Handles NaN in ID columns by defaulting to 0 (not domestic).
    """
    comparison_result = (df['visitor_location_country_id'] == df['prop_country_id'])
    df['is_domestic_travel'] = comparison_result.astype(float).fillna(0).astype(int)
    return df

# New private functions for features previously in preprocessing.add_engineered_features
def _create_historical_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates features based on historical price data."""
    if 'prop_log_historical_price' in df.columns and 'price_usd' in df.columns:
        df['historical_price_usd'] = np.exp(df['prop_log_historical_price'])
        df['price_diff_from_historical_abs'] = df['price_usd'] - df['historical_price_usd']
        df['price_ratio_to_historical'] = df['price_usd'] / df['historical_price_usd'].replace(0, np.nan)
        df['price_ratio_to_historical'] = df['price_ratio_to_historical'].fillna(1)
    return df

def _create_starrating_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates features based on property and visitor star ratings."""
    if ('prop_starrating' in df.columns and 
        'visitor_hist_starrating' in df.columns and 
        'has_visitor_purchase_history' in df.columns):
        df['starrating_diff_from_hist'] = df['prop_starrating'] - df['visitor_hist_starrating']
        df.loc[df['has_visitor_purchase_history'] == 0, 'starrating_diff_from_hist'] = 0
    return df

def _create_review_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates features based on review scores and star ratings."""
    if 'prop_review_score' in df.columns and 'prop_starrating' in df.columns:
        df['review_per_star'] = df['prop_review_score'] / df['prop_starrating'].replace(0, np.nan)
        df['review_per_star'] = df['review_per_star'].fillna(df['prop_review_score'])
    return df

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps to the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame: DataFrame with engineered features.
    """
    print("Starting feature engineering...")
    df_engineered = df.copy()

    # Existing features
    df_engineered = _create_price_rank_feature(df_engineered)
    df_engineered = _create_value_for_money_score(df_engineered)
    df_engineered = _create_domestic_travel_feature(df_engineered)

    # Newly integrated features
    df_engineered = _create_historical_price_features(df_engineered)
    df_engineered = _create_starrating_features(df_engineered)
    df_engineered = _create_review_score_features(df_engineered)

    print(f"Feature engineering complete. Example cols: {df_engineered.columns[:10].tolist()}...")
    return df_engineered