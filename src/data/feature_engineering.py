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

def _add_avg_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates average numeric features per prop_id and merges them back.

    Features averaged:
    - price_usd
    - prop_starrating
    - prop_review_score
    - prop_location_score1
    - prop_location_score2
    - prop_log_historical_price
    - srch_query_affinity_score
    - promotion_flag (becomes proportion)
    """
    print("Adding average property features...")
    features_to_average = [
        'price_usd',
        'prop_starrating',
        'prop_review_score',
        'prop_location_score1',
        'prop_location_score2',
        'prop_log_historical_price',
        'srch_query_affinity_score',
        'promotion_flag'
    ]

    # Create a copy to avoid SettingWithCopyWarning if df is a slice
    df_with_avgs = df.copy()

    for feature in features_to_average:
        if feature in df_with_avgs.columns:
            # Calculate the mean, grouped by prop_id
            avg_feature_name = f'avg_{feature}_for_prop'
            # Ensure prop_id is suitable as an index for transform or for merge
            # Use transform to broadcast the mean back to the original DataFrame shape
            try:
                # Check for potential all-NaN slices if a prop_id has only NaNs for a feature
                # Group by prop_id and calculate mean for the current feature
                grouped = df_with_avgs.groupby('prop_id')[feature]
                
                # Check if all values in any group are NaN, which would make mean NaN
                # Transform is generally robust to this, but direct mean might need check
                # For simplicity, we rely on transform's behavior or direct merge
                
                averages = grouped.transform('mean')
                df_with_avgs[avg_feature_name] = averages
                print(f"  Added {avg_feature_name}")

            except Exception as e:
                print(f"  Could not compute or merge average for {feature}. Error: {e}")
        else:
            print(f"  Feature {feature} not found in DataFrame, skipping average calculation.")
            
    return df_with_avgs

def _add_stddev_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates standard deviation of numeric features per prop_id and merges them back.
    Same list of features as for averages.
    """
    print("Adding stddev property features...")
    features_to_aggregate = [
        'price_usd',
        'prop_starrating',
        'prop_review_score',
        'prop_location_score1',
        'prop_location_score2',
        'prop_log_historical_price',
        'srch_query_affinity_score',
        'promotion_flag' # Stddev of a 0/1 flag can indicate variability in promotion
    ]
    df_with_stddevs = df.copy()

    for feature in features_to_aggregate:
        if feature in df_with_stddevs.columns:
            stddev_feature_name = f'stddev_{feature}_for_prop'
            try:
                grouped = df_with_stddevs.groupby('prop_id')[feature]
                stddevs = grouped.transform('std')
                df_with_stddevs[stddev_feature_name] = stddevs
                # Stddev can be NaN if a group has only one member or all members are identical.
                # Fill NaN stddevs with 0, assuming no variability if cannot be calculated or only one data point.
                df_with_stddevs[stddev_feature_name] = df_with_stddevs[stddev_feature_name].fillna(0)
                print(f"  Added {stddev_feature_name}")
            except Exception as e:
                print(f"  Could not compute or merge stddev for {feature}. Error: {e}")
        else:
            print(f"  Feature {feature} not found, skipping stddev calculation.")
    return df_with_stddevs

def _add_median_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates median of numeric features per prop_id and merges them back.
    Same list of features as for averages.
    """
    print("Adding median property features...")
    features_to_aggregate = [
        'price_usd',
        'prop_starrating',
        'prop_review_score',
        'prop_location_score1',
        'prop_location_score2',
        'prop_log_historical_price',
        'srch_query_affinity_score',
        'promotion_flag'
    ]
    df_with_medians = df.copy()

    for feature in features_to_aggregate:
        if feature in df_with_medians.columns:
            median_feature_name = f'median_{feature}_for_prop'
            try:
                grouped = df_with_medians.groupby('prop_id')[feature]
                medians = grouped.transform('median')
                df_with_medians[median_feature_name] = medians
                print(f"  Added {median_feature_name}")
            except Exception as e:
                print(f"  Could not compute or merge median for {feature}. Error: {e}")
        else:
            print(f"  Feature {feature} not found, skipping median calculation.")
    return df_with_medians

def _add_avg_search_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates average of property-related features per srch_id and merges them back.
    Also adds a count of properties per search.
    """
    print("Adding average search context features (grouped by srch_id)...")
    df_with_search_ctx = df.copy()

    # Count of options in search
    if 'srch_id' in df_with_search_ctx.columns and 'prop_id' in df_with_search_ctx.columns:
        try:
            df_with_search_ctx['num_options_in_search'] = df_with_search_ctx.groupby('srch_id')['prop_id'].transform('size')
            print("  Added num_options_in_search")
        except Exception as e:
            print(f"  Could not compute num_options_in_search. Error: {e}")
    else:
        print("  Skipping num_options_in_search: srch_id or prop_id column missing.")

    features_to_average_by_search = [
        'price_usd',
        'prop_starrating',
        'prop_review_score',
        'prop_location_score1',
        'prop_location_score2',
        'prop_log_historical_price',
        'srch_query_affinity_score',
        'promotion_flag', # Average of 0/1 flag gives proportion
        'prop_brand_bool',  # Average of 0/1 flag gives proportion
        'orig_destination_distance'
    ]

    for feature in features_to_average_by_search:
        if feature in df_with_search_ctx.columns:
            avg_feature_name = f'avg_{feature}_in_search'
            try:
                # Use transform to broadcast the mean back to the original DataFrame shape
                averages = df_with_search_ctx.groupby('srch_id')[feature].transform('mean')
                df_with_search_ctx[avg_feature_name] = averages
                print(f"  Added {avg_feature_name}")
            except Exception as e:
                print(f"  Could not compute or merge average for {feature} in search. Error: {e}")
        else:
            print(f"  Feature {feature} not found, skipping its average calculation for search context.")
            
    return df_with_search_ctx

def _add_stddev_search_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates standard deviation of property-related features per srch_id.
    """
    print("Adding stddev search context features (grouped by srch_id)...")
    df_with_search_stddev = df.copy()
    features_to_aggregate_by_search = [
        'price_usd',
        'prop_starrating',
        'prop_review_score',
        'prop_location_score1',
        'prop_location_score2',
        'prop_log_historical_price',
        'srch_query_affinity_score',
        'promotion_flag',
        'prop_brand_bool',
        'orig_destination_distance'
    ]
    for feature in features_to_aggregate_by_search:
        if feature in df_with_search_stddev.columns:
            agg_feature_name = f'stddev_{feature}_in_search'
            try:
                aggregates = df_with_search_stddev.groupby('srch_id')[feature].transform('std')
                df_with_search_stddev[agg_feature_name] = aggregates.fillna(0) # Fill NaN stddevs (e.g. single item groups)
                print(f"  Added {agg_feature_name}")
            except Exception as e:
                print(f"  Could not compute stddev for {feature} in search. Error: {e}")
        else:
            print(f"  Feature {feature} not found, skipping stddev for search context.")
    return df_with_search_stddev

def _add_median_search_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates median of property-related features per srch_id.
    """
    print("Adding median search context features (grouped by srch_id)...")
    df_with_search_median = df.copy()
    features_to_aggregate_by_search = [
        'price_usd',
        'prop_starrating',
        'prop_review_score',
        'prop_location_score1',
        'prop_location_score2',
        'prop_log_historical_price',
        'srch_query_affinity_score',
        'promotion_flag',
        'prop_brand_bool',
        'orig_destination_distance'
    ]
    for feature in features_to_aggregate_by_search:
        if feature in df_with_search_median.columns:
            agg_feature_name = f'median_{feature}_in_search'
            try:
                aggregates = df_with_search_median.groupby('srch_id')[feature].transform('median')
                df_with_search_median[agg_feature_name] = aggregates
                print(f"  Added {agg_feature_name}")
            except Exception as e:
                print(f"  Could not compute median for {feature} in search. Error: {e}")
        else:
            print(f"  Feature {feature} not found, skipping median for search context.")
    return df_with_search_median

def _add_min_search_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates min of property-related features per srch_id.
    """
    print("Adding min search context features (grouped by srch_id)...")
    df_with_search_min = df.copy()
    features_to_aggregate_by_search = [
        'price_usd',
        'prop_starrating',
        'prop_review_score',
        'prop_location_score1',
        'prop_location_score2',
        'prop_log_historical_price',
        'srch_query_affinity_score',
        'promotion_flag',
        'prop_brand_bool',
        'orig_destination_distance'
    ]
    for feature in features_to_aggregate_by_search:
        if feature in df_with_search_min.columns:
            agg_feature_name = f'min_{feature}_in_search'
            try:
                aggregates = df_with_search_min.groupby('srch_id')[feature].transform('min')
                df_with_search_min[agg_feature_name] = aggregates
                print(f"  Added {agg_feature_name}")
            except Exception as e:
                print(f"  Could not compute min for {feature} in search. Error: {e}")
        else:
            print(f"  Feature {feature} not found, skipping min for search context.")
    return df_with_search_min

def _add_max_search_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates max of property-related features per srch_id.
    """
    print("Adding max search context features (grouped by srch_id)...")
    df_with_search_max = df.copy()
    features_to_aggregate_by_search = [
        'price_usd',
        'prop_starrating',
        'prop_review_score',
        'prop_location_score1',
        'prop_location_score2',
        'prop_log_historical_price',
        'srch_query_affinity_score',
        'promotion_flag',
        'prop_brand_bool',
        'orig_destination_distance'
    ]
    for feature in features_to_aggregate_by_search:
        if feature in df_with_search_max.columns:
            agg_feature_name = f'max_{feature}_in_search'
            try:
                aggregates = df_with_search_max.groupby('srch_id')[feature].transform('max')
                df_with_search_max[agg_feature_name] = aggregates
                print(f"  Added {agg_feature_name}")
            except Exception as e:
                print(f"  Could not compute max for {feature} in search. Error: {e}")
        else:
            print(f"  Feature {feature} not found, skipping max for search context.")
    return df_with_search_max

# --- New Contextual Features ---

def _add_price_relative_to_search_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates price relative to the average price in the search context (srch_id).
    - price_deviation_from_search_avg = price_usd - avg_price_in_search
    - price_ratio_to_search_avg = price_usd / avg_price_in_search
    """
    print("Adding price relative to search average features...")
    df_context = df.copy()
    
    if 'price_usd' in df_context.columns and 'avg_price_usd_in_search' in df_context.columns:
        df_context['price_deviation_from_search_avg'] = df_context['price_usd'] - df_context['avg_price_usd_in_search']
        
        # Handle division by zero or NaN in avg_price_usd_in_search
        df_context['price_ratio_to_search_avg'] = np.where(
            (df_context['avg_price_usd_in_search'].notna()) & (df_context['avg_price_usd_in_search'] != 0),
            df_context['price_usd'] / df_context['avg_price_usd_in_search'],
            np.nan # Or some other placeholder like 1.0 if price_usd is also 0, or if avg is 0
        )
        # If price_usd is 0 and avg_price_usd_in_search is 0, ratio could be 1 (same) or 0.
        # If price_usd > 0 and avg_price_usd_in_search is 0, could be a large number or np.inf.
        # Current np.nan is a safe default.
        print("  Added price_deviation_from_search_avg, price_ratio_to_search_avg")
    else:
        print("  Skipping price relative to search average: required columns missing (price_usd, avg_price_usd_in_search).")
    return df_context

def _add_starrating_relative_to_search_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates star rating relative to the average star rating in the search context (srch_id).
    - starrating_deviation_from_search_avg = prop_starrating - avg_prop_starrating_in_search
    """
    print("Adding starrating relative to search average feature...")
    df_context = df.copy()

    if 'prop_starrating' in df_context.columns and 'avg_prop_starrating_in_search' in df_context.columns:
        df_context['starrating_deviation_from_search_avg'] = df_context['prop_starrating'] - df_context['avg_prop_starrating_in_search']
        print("  Added starrating_deviation_from_search_avg")
    else:
        print("  Skipping starrating relative to search average: required columns missing (prop_starrating, avg_prop_starrating_in_search).")
    return df_context

def _add_price_relative_to_prop_hist_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates price relative to the property's own average historical price.
    - price_deviation_from_prop_hist_avg = price_usd - avg_price_usd_for_prop
    - price_ratio_to_prop_hist_avg = price_usd / avg_price_usd_for_prop
    """
    print("Adding price relative to property's historical average features...")
    df_context = df.copy()

    if 'price_usd' in df_context.columns and 'avg_price_usd_for_prop' in df_context.columns:
        df_context['price_deviation_from_prop_hist_avg'] = df_context['price_usd'] - df_context['avg_price_usd_for_prop']
        
        df_context['price_ratio_to_prop_hist_avg'] = np.where(
            (df_context['avg_price_usd_for_prop'].notna()) & (df_context['avg_price_usd_for_prop'] != 0),
            df_context['price_usd'] / df_context['avg_price_usd_for_prop'],
            np.nan 
        )
        print("  Added price_deviation_from_prop_hist_avg, price_ratio_to_prop_hist_avg")
    else:
        print("  Skipping price relative to property historical average: required columns missing (price_usd, avg_price_usd_for_prop).")
    return df_context

def _add_normalized_price_in_search(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates normalized price within the search result set (srch_id).
    - normalized_price_in_search = (price_usd - min_price_in_search) / (max_price_in_search - min_price_in_search)
    """
    print("Adding normalized price in search feature...")
    df_context = df.copy()

    required_cols = ['price_usd', 'min_price_usd_in_search', 'max_price_usd_in_search']
    if all(col in df_context.columns for col in required_cols):
        denominator = df_context['max_price_usd_in_search'] - df_context['min_price_usd_in_search']
        
        df_context['normalized_price_in_search'] = np.where(
            (denominator.notna()) & (denominator != 0),
            (df_context['price_usd'] - df_context['min_price_usd_in_search']) / denominator,
            0.5 # Or np.nan or 0; 0.5 if min_price=max_price (e.g., only one item or all same price)
        )
        # If min_price == max_price (denominator is 0), all items have same price relative to range.
        # Setting to 0.5 implies it's in the middle of a non-existent range.
        # Can also fill with 0 if only one item.
        print("  Added normalized_price_in_search")
    else:
        print(f"  Skipping normalized price in search: required columns missing. Need: {required_cols}")
    return df_context

# --- End New Contextual Features ---

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

    # Add average property features
    df_engineered = _add_avg_property_features(df_engineered)

    # Add stddev property features (can be commented out for testing)
    df_engineered = _add_stddev_property_features(df_engineered)

    # Add median property features (can be commented out for testing)
    df_engineered = _add_median_property_features(df_engineered)

    # Add average search context features (can be commented out for testing)
    df_engineered = _add_avg_search_context_features(df_engineered)

    # Add other search context aggregations (can be commented out for testing)
    df_engineered = _add_stddev_search_context_features(df_engineered)
    df_engineered = _add_median_search_context_features(df_engineered)
    # df_engineered = _add_min_search_context_features(df_engineered)
    # df_engineered = _add_max_search_context_features(df_engineered)

    # --- Apply New Contextual Features ---
    print("Applying new contextual features...")
    df_engineered = _add_price_relative_to_search_avg(df_engineered)
    df_engineered = _add_starrating_relative_to_search_avg(df_engineered)
    df_engineered = _add_price_relative_to_prop_hist_avg(df_engineered)
    df_engineered = _add_normalized_price_in_search(df_engineered)
    # --- End Apply New Contextual Features ---

    print(f"Feature engineering complete. Example cols: {df_engineered.columns[:15].tolist()}...") # Adjusted to show more cols
    return df_engineered