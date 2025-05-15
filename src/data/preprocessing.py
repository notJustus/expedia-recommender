import pandas as pd
import numpy as np # Added for np.exp

def handle_missing_visitor_hist_starrating(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'visitor_hist_starrating' feature."""
    df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(0)
    return df

def handle_missing_visitor_hist_adr_usd(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'visitor_hist_adr_usd' feature."""
    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(0)
    return df

def handle_missing_prop_review_score(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'prop_review_score' feature."""
    df['prop_review_score'] = df['prop_review_score'].fillna(0)
    return df

def handle_missing_prop_location_score2(df: pd.DataFrame, median_val_override: float | None = None) -> pd.DataFrame:
    """Handles missing values for the 'prop_location_score2' feature."""
    if median_val_override is not None:
        fill_val = median_val_override
    else:
        fill_val = df['prop_location_score2'].median()
    df['prop_location_score2'] = df['prop_location_score2'].fillna(fill_val)
    return df

def handle_missing_srch_query_affinity_score(df: pd.DataFrame, min_val_override: float | None = None) -> pd.DataFrame:
    """Handles missing values for the 'srch_query_affinity_score' feature."""
    if min_val_override is not None:
        fill_value = min_val_override - 100 # Consistent with original logic if min_val is from train
    else:
        min_affinity_score_df = df['srch_query_affinity_score'].min()
        fill_value = min_affinity_score_df - 100 if pd.notna(min_affinity_score_df) else -999
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(fill_value)
    return df

def handle_missing_orig_destination_distance(df: pd.DataFrame, median_val_override: float | None = None) -> pd.DataFrame:
    """Handles missing values for the 'orig_destination_distance' feature."""
    if median_val_override is not None:
        fill_val = median_val_override
    else:
        fill_val = df['orig_destination_distance'].median()
    df['orig_destination_distance'] = df['orig_destination_distance'].fillna(fill_val)
    return df

def handle_missing_comp1_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp1_rate' feature."""
    df['comp1_rate'] = df['comp1_rate'].fillna(0)
    return df

def handle_missing_comp1_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp1_inv' feature."""
    df['comp1_inv'] = df['comp1_inv'].fillna(0)
    return df

def handle_missing_comp1_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp1_rate_percent_diff' feature."""
    df['comp1_rate_percent_diff'] = df['comp1_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp2_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp2_rate' feature."""
    df['comp2_rate'] = df['comp2_rate'].fillna(0)
    return df

def handle_missing_comp2_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp2_inv' feature."""
    df['comp2_inv'] = df['comp2_inv'].fillna(0)
    return df

def handle_missing_comp2_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp2_rate_percent_diff' feature."""
    df['comp2_rate_percent_diff'] = df['comp2_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp3_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp3_rate' feature."""
    df['comp3_rate'] = df['comp3_rate'].fillna(0)
    return df

def handle_missing_comp3_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp3_inv' feature."""
    df['comp3_inv'] = df['comp3_inv'].fillna(0)
    return df

def handle_missing_comp3_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp3_rate_percent_diff' feature."""
    df['comp3_rate_percent_diff'] = df['comp3_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp4_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp4_rate' feature."""
    df['comp4_rate'] = df['comp4_rate'].fillna(0)
    return df

def handle_missing_comp4_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp4_inv' feature."""
    df['comp4_inv'] = df['comp4_inv'].fillna(0)
    return df

def handle_missing_comp4_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp4_rate_percent_diff' feature."""
    df['comp4_rate_percent_diff'] = df['comp4_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp5_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp5_rate' feature."""
    df['comp5_rate'] = df['comp5_rate'].fillna(0)
    return df

def handle_missing_comp5_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp5_inv' feature."""
    df['comp5_inv'] = df['comp5_inv'].fillna(0)
    return df

def handle_missing_comp5_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp5_rate_percent_diff' feature."""
    df['comp5_rate_percent_diff'] = df['comp5_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp6_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp6_rate' feature."""
    df['comp6_rate'] = df['comp6_rate'].fillna(0)
    return df

def handle_missing_comp6_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp6_inv' feature."""
    df['comp6_inv'] = df['comp6_inv'].fillna(0)
    return df

def handle_missing_comp6_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp6_rate_percent_diff' feature."""
    df['comp6_rate_percent_diff'] = df['comp6_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp7_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp7_rate' feature."""
    df['comp7_rate'] = df['comp7_rate'].fillna(0)
    return df

def handle_missing_comp7_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp7_inv' feature."""
    df['comp7_inv'] = df['comp7_inv'].fillna(0)
    return df

def handle_missing_comp7_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp7_rate_percent_diff' feature."""
    df['comp7_rate_percent_diff'] = df['comp7_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp8_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp8_rate' feature."""
    df['comp8_rate'] = df['comp8_rate'].fillna(0)
    return df

def handle_missing_comp8_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp8_inv' feature."""
    df['comp8_inv'] = df['comp8_inv'].fillna(0)
    return df

def handle_missing_comp8_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp8_rate_percent_diff' feature."""
    df['comp8_rate_percent_diff'] = df['comp8_rate_percent_diff'].fillna(0)
    return df

def handle_missing_values(df: pd.DataFrame, is_train: bool = True, imputation_values: dict | None = None) -> tuple[pd.DataFrame, dict | None]:
    """
    Handles missing values for all specified features.
    Each feature is handled by its own function.
    If is_train is True, calculates imputation statistics (medians, etc.) and returns them.
    If is_train is False, uses pre-calculated imputation_values.
    """
    df = df.copy()
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])

    calculated_imputation_values = {}

    # Indicator for visitor purchase history
    if 'visitor_hist_starrating' in df.columns:
        df['has_visitor_purchase_history'] = df['visitor_hist_starrating'].notnull().astype(int)
        df = handle_missing_visitor_hist_starrating(df)
    if 'visitor_hist_adr_usd' in df.columns:
        df = handle_missing_visitor_hist_adr_usd(df)
    if 'prop_review_score' in df.columns:
        df = handle_missing_prop_review_score(df)

    # prop_location_score2
    if 'prop_location_score2' in df.columns:
        df['prop_location_score2_is_missing'] = df['prop_location_score2'].isnull().astype(int)
        loc_score2_median = None
        if is_train:
            loc_score2_median = df['prop_location_score2'].median()
            calculated_imputation_values['prop_location_score2_median'] = loc_score2_median
        elif imputation_values:
            loc_score2_median = imputation_values.get('prop_location_score2_median')
        df = handle_missing_prop_location_score2(df, median_val_override=loc_score2_median)

    # srch_query_affinity_score
    if 'srch_query_affinity_score' in df.columns:
        query_affinity_min = None
        if is_train:
            query_affinity_min = df['srch_query_affinity_score'].min()
            calculated_imputation_values['srch_query_affinity_score_min'] = query_affinity_min
        elif imputation_values:
            query_affinity_min = imputation_values.get('srch_query_affinity_score_min')
        df = handle_missing_srch_query_affinity_score(df, min_val_override=query_affinity_min)

    # orig_destination_distance
    if 'orig_destination_distance' in df.columns:
        df['orig_destination_distance_is_missing'] = df['orig_destination_distance'].isnull().astype(int)
        dest_dist_median = None
        if is_train:
            dest_dist_median = df['orig_destination_distance'].median()
            calculated_imputation_values['orig_destination_distance_median'] = dest_dist_median
        elif imputation_values:
            dest_dist_median = imputation_values.get('orig_destination_distance_median')
        df = handle_missing_orig_destination_distance(df, median_val_override=dest_dist_median)

    # Competitor data
    for i in range(1, 9):
        if f'comp{i}_rate' in df.columns: # Check if comp column exists
            df[f'comp{i}_has_data'] = df[f'comp{i}_rate'].notnull().astype(int)
            # Imputation for comp features is fillna(0), no train/test difference needed for the value itself
            df = globals()[f'handle_missing_comp{i}_rate'](df)
            df = globals()[f'handle_missing_comp{i}_inv'](df)
            df = globals()[f'handle_missing_comp{i}_rate_percent_diff'](df)
        else:
            # If the base comp column (e.g. comp_rate) doesn't exist, create the _has_data as 0
            df[f'comp{i}_has_data'] = 0
    
    return df, (calculated_imputation_values if is_train else None)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds new features to the DataFrame."""
    df = df.copy()

    # De-log historical price: prop_log_historical_price = 0 if not sold. 
    # np.exp(0) is 1. So, if it was 0 (not sold), historical price becomes 1.
    # If it was a valid log price, it gets converted back.
    # This might need adjustment if a 0 in log_historical_price truly means $0 price vs. not sold/unknown.
    # Assuming 0 in log scale means a very low actual price or was simply set to 0 for missing.
    if 'prop_log_historical_price' in df.columns and 'price_usd' in df.columns:
        df['historical_price_usd'] = np.exp(df['prop_log_historical_price'])
        # Avoid division by zero or issues with historical_price_usd=0 or 1 (if exp(0))
        # Only calculate diff if historical_price is somewhat realistic (e.g. > 1 after exp)
        df['price_diff_from_historical_abs'] = df['price_usd'] - df['historical_price_usd']
        df['price_ratio_to_historical'] = df['price_usd'] / df['historical_price_usd'].replace(0, np.nan) # Avoid div by zero
        df['price_ratio_to_historical'] = df['price_ratio_to_historical'].fillna(1) # If historical was 0, assume ratio is 1 (no difference)

    if 'prop_starrating' in df.columns and 'visitor_hist_starrating' in df.columns and 'has_visitor_purchase_history' in df.columns:
        # Only calculate diff if there is a history
        df['starrating_diff_from_hist'] = df['prop_starrating'] - df['visitor_hist_starrating']
        # Set to 0 or a neutral value if no history, or handle via missingness of visitor_hist_starrating
        df.loc[df['has_visitor_purchase_history'] == 0, 'starrating_diff_from_hist'] = 0 # Or np.nan and let imputation handle

    # Example: Interaction between prop_review_score and prop_starrating
    if 'prop_review_score' in df.columns and 'prop_starrating' in df.columns:
        # Ensure prop_starrating is not 0 to avoid division by zero if that makes sense
        df['review_per_star'] = df['prop_review_score'] / df['prop_starrating'].replace(0, np.nan)
        df['review_per_star'] = df['review_per_star'].fillna(df['prop_review_score']) # if star is 0, just use review score

    print(f"Engineered features added. Current columns: {df.columns.tolist()[:15]}...") # Print first few
    return df
