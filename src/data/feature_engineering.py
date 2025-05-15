import pandas as pd
import numpy as np


def _add_is_recurring_customer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a binary feature 'is_recurring_customer' to the DataFrame.
    A customer is considered recurring if they have a historical star rating
    or historical average daily rate.

    Args:
        df (pd.DataFrame): Input DataFrame, must contain 
                           'visitor_hist_starrating' and 'visitor_hist_adr_usd'.

    Returns:
        pd.DataFrame: DataFrame with the added 'is_recurring_customer' column.
    """
    if 'visitor_hist_starrating' not in df.columns or 'visitor_hist_adr_usd' not in df.columns:
        raise ValueError("DataFrame must contain 'visitor_hist_starrating' and 'visitor_hist_adr_usd' columns.")
        
    # True if 'visitor_hist_starrating' is not NaN OR 'visitor_hist_adr_usd' is not NaN
    # .notna() returns True for non-NaN values
    condition = df['visitor_hist_starrating'].notna() | df['visitor_hist_adr_usd'].notna()
    
    df['is_recurring_customer'] = condition.astype(int)
    return df


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

# --- New Date & Time Feature Functions ---
def _create_search_hour_of_day_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts the hour of the day from 'date_time'."""
    if 'date_time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date_time']):
        df['search_hour_of_day'] = df['date_time'].dt.hour
    else:
        df['search_hour_of_day'] = np.nan # Or some other default if date_time is missing/not datetime
    return df

def _create_search_day_of_week_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts the day of the week (0=Monday, 6=Sunday) from 'date_time'."""
    if 'date_time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date_time']):
        df['search_day_of_week'] = df['date_time'].dt.dayofweek
    else:
        df['search_day_of_week'] = np.nan
    return df

def _create_search_month_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts the month from 'date_time'."""
    if 'date_time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date_time']):
        df['search_month'] = df['date_time'].dt.month
    else:
        df['search_month'] = np.nan
    return df

def _create_is_weekend_search_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a boolean feature (0 or 1) if search was on a weekend (Sat=5, Sun=6)."""
    if 'date_time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date_time']):
        # dayofweek: Monday=0 to Sunday=6
        df['is_weekend_search'] = df['date_time'].dt.dayofweek.isin([5, 6]).astype(int)
    else:
        df['is_weekend_search'] = 0 # Default to not weekend if no date_time
    return df

def _create_booking_window_weeks_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Converts 'srch_booking_window' (days) into weeks."""
    if 'srch_booking_window' in df.columns:
        df['booking_window_weeks'] = (df['srch_booking_window'] / 7).round().astype(float) # Keep as float for potential NaNs or use int if NaNs are handled
    else:
        df['booking_window_weeks'] = np.nan
    return df

# --- End New Date & Time Feature Functions ---

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

    # Add Date & Time features
    print("Applying date & time features...")
    df_engineered = _create_search_hour_of_day_feature(df_engineered)
    df_engineered = _create_search_day_of_week_feature(df_engineered)
    df_engineered = _create_search_month_feature(df_engineered)
    df_engineered = _create_is_weekend_search_feature(df_engineered)
    df_engineered = _create_booking_window_weeks_feature(df_engineered)

    # User Related Features
    df_engineered = _add_is_recurring_customer(df_engineered)

    print(f"Feature engineering complete. Example cols: {df_engineered.columns[:15].tolist()}...") # Adjusted to show more cols
    return df_engineered